"""m3_batch_core: clean, importable batched-generation core for MiniMax-M3 (mlx-vlm).

Packages the ALREADY-PROVEN algorithm from /tmp/batched_gen.py (Stage 1) and
/tmp/stage2_gen.py (Stage 2) into one reusable function, `generate_batch`.

Memory-safe discipline (do NOT change):
  * left-pad ragged prompts into one batch + a BatchKVCache per layer
  * logits-free prefill: run `inner(inputs=..., cache=..., mask=None)` -> hidden states,
    then `lm_head` on the LAST token ONLY (never lm_head over all positions)
  * lockstep decode with per-row EOS / max_tokens stop
  * `cache.filter(mx.array(keep_positions))` drops finished rows mid-decode
  * `mx.eval` once per step to bound the graph

Dependency-light: mlx.core + mlx_vlm.models.cache only.
"""
import time  # [auditfix] #3: wall-clock cap on the decode loop
import mlx.core as mx
from mlx_vlm.models.cache import BatchKVCache


def _resolve_surface(model):
    """Return (inner, head, num_layers) for both the real mlx-vlm MODEL wrapper and
    the tiny test LanguageModel. Probe with hasattr (no isinstance coupling)."""
    # Real wrapper exposes `.language_model`; the tiny test LanguageModel exposes
    # `.model` / `.lm_head` directly.
    tower = model.language_model if hasattr(model, "language_model") else model
    inner = tower.model        # hidden-state producer (NO lm_head inside)
    head = tower.lm_head       # vocab projection, applied to last token only
    num_layers = len(inner.layers)
    return inner, head, num_layers


def _next_logits_lastpos(inner, head, inputs, cache):
    """Run hidden states for `inputs` and project ONLY the last position -> [B, vocab]."""
    h = inner(inputs=inputs, cache=cache, mask=None)   # [B, T, H], NO lm_head over all T
    return head(h[:, -1, :])                            # lm_head on LAST token only -> [B, vocab]


def _sample(logits, temp, key):
    """Greedy when temp==0, else temperature sampling. Returns ([B] tokens, next_key)."""
    if temp == 0.0:
        return logits.argmax(-1), key
    key, sub = mx.random.split(key)
    toks = mx.random.categorical(logits * (1.0 / temp), key=sub)
    return toks, key


def generate_batch(model, prompt_token_lists, max_tokens_list, eos_ids, temp=0.0, seed=0,
                   cancels=None, max_gen_s=None, clear_cache_steps=0):  # [auditfix] #3/#4/#2
    """Batched generation over ragged prompts (bit-exact vs single-stream for greedy).

    Args:
        model: loaded mlx-vlm model wrapper (real) OR tiny test LanguageModel.
            Real: language tower = model.language_model; inner = .model; head = .lm_head.
            Test: inner = model.model; head = model.lm_head. Probed via hasattr.
        prompt_token_lists: list of per-row prompt token ids (list[int] or mx.array),
            already templated/tokenized.
        max_tokens_list: list[int], max new tokens per row.
        eos_ids: set/list of token ids that stop a row (real model eos = 200020).
        temp: 0.0 => greedy/deterministic; >0 => temperature sampling (mx.random, seeded).
        seed: RNG seed for sampling reproducibility.
        cancels: [auditfix] #4 — optional list (len == B) of zero-arg callables, one per row;
            a row whose callable returns True is dropped at the next decode step (its caller
            timed out — never keep generating for a dead caller). None = never cancel.
        max_gen_s: [auditfix] #3 — optional wall-clock cap (seconds) for the WHOLE batched
            decode; when exceeded, all rows stop after the in-flight step (finish = length,
            since the last token is normally not EOS). None = uncapped (old behavior).
        clear_cache_steps: [auditfix] #2 — if > 0, call mx.clear_cache() every N decode steps
            to trim Metal buffer-cache regrowth on long gens. 0 = off (old behavior).

    Returns:
        list[list[int]] — generated token ids per row (NOT including the prompt). A stopping
        EOS token IS included in the row (matches the proven Stage-2 reference); generation
        halts at EOS or the row's max_tokens.
    """
    inner, head, num_layers = _resolve_surface(model)

    B = len(prompt_token_lists)
    outs = [[] for _ in range(B)]
    if B == 0:
        return outs

    # Normalize prompts to mx.array (int ids).
    prompts = [p if isinstance(p, mx.array) else mx.array(p) for p in prompt_token_lists]
    eos_set = set(int(e) for e in eos_ids)

    # Rows that should emit nothing (max_tokens <= 0) are excluded up front so their KV
    # never enters the batch; they keep their empty output.
    active = [b for b in range(B) if max_tokens_list[b] > 0]
    if not active:
        return outs

    # Left-pad the (active) ragged prompts into one [B', Lmax] batch.
    Lmax = max(len(prompts[b]) for b in active)
    if Lmax == 0:
        # Every active prompt is empty -> nothing to prefill from. Degenerate; return empties.
        return outs
    lp = [Lmax - len(prompts[b]) for b in active]
    ids = mx.stack([
        mx.concatenate([mx.zeros((lp[i],), dtype=prompts[orig].dtype), prompts[orig]])
        for i, orig in enumerate(active)
    ])

    cache = [BatchKVCache(lp) for _ in range(num_layers)]
    key = mx.random.key(seed)

    # Logits-free prefill (lm_head on last token only).
    logits = _next_logits_lastpos(inner, head, ids, cache)
    mx.eval(logits)

    t_start = time.time(); step = 0  # [auditfix] #3
    while active:
        toks, key = _sample(logits, temp, key)
        mx.eval(toks)
        step += 1
        if clear_cache_steps and (step % clear_cache_steps) == 0:  # [auditfix] #2 (opt-in)
            mx.clear_cache()
        # [auditfix] #3: whole-batch wall-clock cap — a runaway gen must not hold the single
        # worker (and GEN_LOCK) indefinitely. Checked once per lockstep decode step.
        timed_out = max_gen_s is not None and (time.time() - t_start) > max_gen_s
        keep = []  # positions (into current `active`) of rows that continue
        for pos, orig in enumerate(active):
            t = int(toks[pos].item())
            outs[orig].append(t)
            # [auditfix] #4: a cancelled row (dead caller) stops exactly like EOS/max_tokens
            cancelled = cancels is not None and cancels[orig] is not None and cancels[orig]()
            if not (t in eos_set or len(outs[orig]) >= max_tokens_list[orig]
                    or timed_out or cancelled):  # [auditfix] #3/#4
                keep.append(pos)
        if not keep:
            break
        if len(keep) != len(active):
            ka = mx.array(keep)
            for c in cache:
                c.filter(ka)              # drop finished rows' KV, preserve survivors' offsets
            active = [active[p] for p in keep]
            toks = toks[ka]
        logits = _next_logits_lastpos(inner, head, toks[:, None], cache)
        mx.eval(logits)

    return outs


# --------------------------------------------------------------------------------------
# Unit tests: tiny random-weight MiniMax-M3 LanguageModel (same config as /tmp/batched_gen.py).
# Run: python ~/m3_batch_core.py
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from mlx_vlm.models.minimax_m3.config import TextConfig
    from mlx_vlm.models.minimax_m3.language import LanguageModel

    cfg = TextConfig(
        hidden_size=128, intermediate_size=64, dense_intermediate_size=128,
        shared_intermediate_size=64, num_hidden_layers=2, num_attention_heads=64,
        num_key_value_heads=4, head_dim=128, vocab_size=128, rotary_dim=64,
        num_local_experts=4, num_experts_per_tok=2, moe_layer_freq=[1, 1],
        sparse_attention_config={"use_sparse_attention": True, "sparse_attention_freq": [0, 0]},
        tie_word_embeddings=False)
    lm = LanguageModel(cfg); lm.set_dtype(mx.float16); mx.eval(lm.parameters())
    NL = cfg.num_hidden_layers
    inner, head = lm.model, lm.lm_head

    # Reference single-stream paths (logits-free) — independent of generate_batch.
    def _pf(ids, cache):
        return head(inner(inputs=ids, cache=cache, mask=None)[:, -1, :])

    def _dc(tok, cache):
        return head(inner(inputs=tok, cache=cache, mask=None)[:, -1, :])

    def single_greedy(p, n):
        cache = [BatchKVCache([0]) for _ in range(NL)]
        logit = _pf(p[None], cache); mx.eval(logit)
        out = []
        for _ in range(n):
            t = int(logit.argmax(-1)[0].item()); out.append(t)
            logit = _dc(mx.array([[t]]), cache); mx.eval(logit)
        return out

    def single_stop(p, mt, eos):
        cache = [BatchKVCache([0]) for _ in range(NL)]
        logit = _pf(p[None], cache); mx.eval(logit)
        out = []
        for _ in range(mt):
            t = int(logit.argmax(-1)[0].item()); out.append(t)
            if t == eos:
                break
            logit = _dc(mx.array([[t]]), cache); mx.eval(logit)
        return out

    all_pass = True

    # --- Test 1: greedy parity vs single-stream (ragged, bit-identical) ---
    mx.random.seed(3)
    prompts = [mx.random.randint(1, cfg.vocab_size, (l,)) for l in [30, 18, 24, 9]]
    N = 12
    batched = generate_batch(lm, prompts, [N] * len(prompts), eos_ids={-1}, temp=0.0)
    t1 = True
    for b in range(len(prompts)):
        s = single_greedy(prompts[b], N)
        m = batched[b] == s
        t1 = t1 and m
        print("  T1 row%d len=%2d match=%s batched=%s" % (b, len(prompts[b]), m, batched[b][:6]))
        if not m:
            print("        single =%s" % s[:6])
    print("TEST 1 greedy parity: %s\n" % ("PASS" if t1 else "FAIL"))
    all_pass = all_pass and t1

    # --- Test 2: per-row EOS + heterogeneous max_tokens stop (mirror stage2) ---
    mx.random.seed(7)
    prompts2 = [mx.random.randint(1, cfg.vocab_size, (l,)) for l in [30, 18, 24, 12]]
    max_toks = [15, 6, 20, 9]
    eos = 29
    batched2 = generate_batch(lm, prompts2, max_toks, eos_ids={eos}, temp=0.0)
    t2 = True
    for i in range(len(prompts2)):
        s = single_stop(prompts2[i], max_toks[i], eos)
        m = batched2[i] == s
        t2 = t2 and m
        print("  T2 row%d mt=%2d stopped@%d match=%s%s"
              % (i, max_toks[i], len(batched2[i]), m,
                 "" if m else (" B=%s S=%s" % (batched2[i], s))))
    print("TEST 2 per-row stop + filter: %s\n" % ("PASS" if t2 else "FAIL"))
    all_pass = all_pass and t2

    # --- Test 3: temperature sampling determinism (same seed -> identical) ---
    mx.random.seed(11)
    prompts3 = [mx.random.randint(1, cfg.vocab_size, (l,)) for l in [20, 14, 27]]
    runA = generate_batch(lm, prompts3, [10] * 3, eos_ids={-1}, temp=0.8, seed=123)
    runB = generate_batch(lm, prompts3, [10] * 3, eos_ids={-1}, temp=0.8, seed=123)
    t3 = runA == runB
    for i in range(len(prompts3)):
        print("  T3 row%d sampled=%s identical=%s" % (i, runA[i][:6], runA[i] == runB[i]))
    print("TEST 3 temp-sampling determinism: %s\n" % ("PASS" if t3 else "FAIL"))
    all_pass = all_pass and t3

    # --- Edge cases: B=1 clean fall-through, empty-prompt row, max_tokens=0 row ---
    e1 = generate_batch(lm, [mx.random.randint(1, cfg.vocab_size, (16,))], [5], eos_ids={-1}, temp=0.0)
    e1_ok = len(e1) == 1 and len(e1[0]) == 5
    # empty prompt row alongside a real one: must not crash, empty row may yield tokens via pad
    e2 = generate_batch(lm, [mx.array([], dtype=mx.int32), prompts[0]], [4, 4], eos_ids={-1}, temp=0.0)
    e2_ok = len(e2) == 2 and len(e2[1]) == 4
    # max_tokens=0 row -> empty output, sibling unaffected
    e3 = generate_batch(lm, [prompts[0], prompts[1]], [0, 3], eos_ids={-1}, temp=0.0)
    e3_ok = e3[0] == [] and len(e3[1]) == 3
    te = e1_ok and e2_ok and e3_ok
    print("  EDGE B=1=%s empty-prompt=%s maxtok0=%s" % (e1_ok, e2_ok, e3_ok))
    print("EDGE CASES: %s\n" % ("PASS" if te else "FAIL"))
    all_pass = all_pass and te

    print("=== OVERALL: %s ===" % ("ALL PASS" if all_pass else "FAIL"))
    import sys as _sys
    _sys.exit(0 if all_pass else 1)
