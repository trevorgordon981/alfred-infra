#!/usr/bin/env python3
"""Hold MiniMax-M3 8-bit resident and serve it two ways:
  1) Legacy test format:  POST / {"prompt"(str|messages),"max_tokens","temp","thinking","save_to","tools"}
       -> {"text","tokens","prompt_tokens","prompt_tps","tps","finish"}
  2) OpenAI-compatible:   POST /v1/chat/completions {"model","messages","max_tokens","temperature","tools","thinking","stream"}
       -> standard chat.completion envelope (choices[].message.content). This is the PROD path the
          trading stack (gateway/trader/daily_recommend) and Slack-Alfred talk to.
GET / -> "ready"; GET /v1/models -> OpenAI model list (health/up-checks).
Thinking defaults OFF (mandatory-thinking is what stalled M2.7 on the breakout brief); opt in per-request
via a leading "think:" on the last user message, or an explicit {"thinking":"enabled"} field."""
import json, contextlib, io, time, os, threading, re, base64, tempfile, sys, traceback, random, hmac, stat, math  # [auditfix] +sys/traceback (#5), +random (#10)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Keep the module importable for request/security unit tests on machines that do not have the
# 300GB-serving MLX environment. Production still fails fast in main() before binding a socket if
# any runtime dependency is unavailable; generation behavior is unchanged when they are present.
_RUNTIME_IMPORT_ERROR = None
try:
    import mlx.core as mx  # [auditfix] #2/#3: clear_cache after gens + synchronize on cap-abort (mirror glm_serve_batched)
    from mlx_vlm import load, generate, stream_generate
    from mlx_vlm.generate.dispatch import PromptCacheState
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
except Exception as _e:
    _RUNTIME_IMPORT_ERROR = _e
    mx = load = generate = stream_generate = apply_chat_template = load_config = None

    class PromptCacheState:  # inert placeholder; no generation path is reachable without runtime
        pass

# --- constrained JSON decoding (increment-1 #1): the serving mlx_vlm ships an llguidance-backed,
# batch-aware, think-block-aware structured decoder. When a request carries response_format, we
# build a per-request logits processor and run it SOLO (the batch core has no logits-processor
# hook yet) so the trader's JSON is GRAMMAR-GUARANTEED valid, not best-effort. M3 uses <mm:think>
# tags, so the ThinkingAware wrapper is told the M3 end token. Import remains soft so ordinary text
# serving can start without the optional stack, but an explicit response_format then fails closed
# with 503 instead of silently running unconstrained.
if _RUNTIME_IMPORT_ERROR is None:
    try:
        from mlx_vlm.structured import build_json_schema_logits_processor, ThinkingAwareLogitsProcessor
        _STRUCTURED_OK = True
    except Exception as _e:
        _STRUCTURED_OK = False
        print("[warn] structured decoding unavailable (%s); response_format will be unconstrained" % _e, flush=True)
else:
    _STRUCTURED_OK = False
    build_json_schema_logits_processor = ThinkingAwareLogitsProcessor = None


def _log_exc(where):
    # [auditfix] #5: surface exceptions to STDERR (-> err.log via launchd StandardErrorPath).
    # 500s were previously swallowed into the HTTP body (str(e)) and never captured anywhere
    # (err.log was empty), so real failure causes were invisible. Mirror of glm_serve_batched.
    print("[ERROR] %s @ %s" % (where, time.strftime("%Y-%m-%dT%H:%M:%S")),
          file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()


def _log_error(where, message):
    """Log a server-side error without requiring or exposing an exception string."""
    print("[ERROR] %s @ %s: %s" % (where, time.strftime("%Y-%m-%dT%H:%M:%S"), message),
          file=sys.stderr, flush=True)

# Default = original prod weights (the instant-rollback target). PROD may override the served
# weights via M3_MODEL_DIR in the launchd plist (2026-06-23 retune cutover): set it to repoint
# :8082, unset/revert to roll back to MiniMax-M3-8bit. MODEL_ID stays "minimax-m3" so the
# gateway/trader/chat path and the watchdog /v1/models id-check are unchanged across the swap.
MD = os.path.realpath(os.path.expanduser(
    os.environ.get("M3_MODEL_DIR", "~/models/MiniMax-M3-8bit")))
MODEL_ID = "minimax-m3"
PORT = int(os.environ.get("M3_PORT", "8085"))  # nightly eval: 8085; PROD sets M3_PORT=8082

# WEDGE FIX (2026-06-22): bind the listening socket BEFORE loading the 256GB of weights, so the
# port LISTENs in milliseconds and the watchdog can always see a live socket. Real inference is
# gated behind READY (503 "loading" until the load finishes). If the load OOM-dies mid-flight we
# os._exit(1) so launchd KeepAlive relaunches cleanly instead of leaving "weights resident, no
# listener, process alive" — the un-healable wedge. MODEL/PROC/CFG are populated by _load() below.
READY = False
MODEL = PROC = CFG = None

# Threaded server (connections accepted instantly so callers' connect-timeouts don't
# trip a fallback) but GPU generation is serialized — the model/tokenizer are NOT
# thread-safe, so only one generate() runs at a time; others queue on the lock.
GEN_LOCK = threading.Lock()
# QUEUE-NOT-SHED (2026-06-23): a request that cannot grab the GPU now WAITS its turn (FIFO on the
# lock) instead of 503-ing after a short window. Default raised 45 -> 600s so the live trader and
# the (thinking-on, multi-minute) daily slate QUEUE behind each other rather than collide into 503
# -> retry/backoff/broken-fallback churn. Single-generation safety is unchanged (still ONE gen at a
# time -- this only lengthens how long a queued caller is willing to wait). We still 503 if this
# generous bound is genuinely exceeded (should be rare); the watchdog v4 still backstops a true wedge.
# Prior history: 8->45 on 2026-06-13 (Slack-Alfred ~5 parallel calls/message were 503-storming).
LOCK_WAIT_S = int(os.environ.get("M3_LOCK_WAIT_S", "600"))
# [auditfix] #3: per-generation wall-clock cap (port of glm_serve_batched MAX_GEN_S). Default 600s
# == LOCK_WAIT_S on purpose: past that point every queued caller has already timed out (Busy), so
# finishing the gen is pure waste that just holds the single worker. Legit multi-minute
# thinking-on slate gens still fit under 600s (default-safe); tune DOWN via env after live
# validation. Enforced on the streamed prefix-cache solo path and the batched decode loop; the
# fresh-cache retry / vision generate() fallbacks are single blocking calls and cannot be
# interrupted mid-flight (documented limitation).
MAX_GEN_S = int(os.environ.get("M3_MAX_GEN_S", "600"))
# [auditfix] #2: optional periodic mx.clear_cache() every N decode steps during LONG gens
# (batched loop + streamed solo loop), for the GLM-style file-cache/Metal-buffer regrowth
# failure mode. 0 = OFF (default: opt-in — mid-decode clears trade a little throughput for
# memory hygiene; the always-on `finally` clear below is the default-safe half of this fix).
CLEAR_CACHE_STEPS = int(os.environ.get("M3_CLEAR_CACHE_STEPS", "0"))
# [inc4a] chunked prefill: bound the peak prefill allocation on the solo stream_generate path (OOM
# guard for long prompts). stream_generate forwards prefill_step_size to generate_step, which then
# processes the prompt in <=N-token chunks instead of one giant forward. Default 2048 is a safe
# always-on value. NOTE (deferred): we do NOT yield GEN_LOCK between prefill chunks — the library
# does not expose chunk boundaries to the caller (prefill happens inside stream_generate's first
# next()), so there is no hook to release/re-acquire the lock mid-prefill. Single-generation safety
# is unaffected; a long prefill still holds the lock for its full duration, same as today.
PREFILL_STEP = int(os.environ.get("M3_PREFILL_STEP", "2048"))


class Busy(Exception):
    """Raised when the GPU lock can't be acquired in time. Callers return 503 so the
    gateway falls back instead of a request backlog snowballing (the 2026-06-13 wedge)."""


# Prefix caching (validated correct on M3 block-sparse path 2026-06-13: cold==cached output,
# 25x TTFT on a 3.3k prompt; KV-quant also clean). Reuse the KV cache across consecutive turns
# of the same conversation -- stream_generate(prompt_cache_state=...) trims to the shared token
# prefix and only prefills the new suffix. ONE shared state is correct because GEN_LOCK
# serializes all generation; a request that does not share a prefix just rebuilds (still
# correct, no speedup that turn). Memory cost = one persistent KV cache (~GBs), not N, so no
# OOM risk. Disable instantly with env M3_PREFIX_CACHE=0 (falls back to fresh-cache generate()).
# [audit H3] DEFAULT FLIPPED 1 -> 0. The single shared PCSTATE poisons cross-conversation requests
# (the 2026-07-09 0.3 tok/s incident). Defaulting it ON meant any env that turned APC off silently
# re-armed the footgun. Now: no cache unless a mechanism is EXPLICITLY enabled. APC (keyed, safe)
# is the intended path; legacy PCSTATE stays available but off-by-default and warns loudly if used.
PREFIX_CACHE = os.environ.get("M3_PREFIX_CACHE", "0") != "0"
PCSTATE = PromptCacheState()
if PREFIX_CACHE:
    print("[warn] M3_PREFIX_CACHE=1: using the SINGLE-SHARED PromptCacheState (cross-conversation "
          "pollution risk — the 0.3 tok/s incident). Prefer M3_APC=1 (keyed) instead.", flush=True)

# APC (Automatic Prefix Cache): the library's CONTENT-HASH-KEYED, block-based, LRU prefix
# cache. Unlike the single shared PCSTATE (which one conversation's KV pollutes for the next
# unrelated request -> the 2026-07-09 0.3 tok/s incident), APC keys blocks by prompt-content
# hash, so byron's identical ~800-tok system prompt and Alfred's long history live in SEPARATE
# blocks and never poison each other. Enabled with M3_APC=1 (default OFF -> unchanged prod
# behavior). num_blocks * 16 tok = cache capacity; 1024 blocks = ~16K tokens (byron system
# prompt is ~50 blocks, reused every call). APC has its own RAM guards + optional disk spill.
M3_APC = os.environ.get("M3_APC", "0") != "0"
APC = None
if M3_APC:
    from mlx_vlm import apc as _apc_mod
    # [audit C3] default 1024 -> 4096 blocks (~65K tokens). 1024 (16K tok) could NOT hold Alfred's
    # ~38K-token history, so an Alfred request evicted byron's hot ~800-tok system prefix AND got
    # no reuse itself. 4096 blocks fits one full Alfred conversation (~2375 blk) + byron's system
    # prefix (~66 blk) + Slack burst headroom. ~4-8GB of KV against this box's headroom; APC's own
    # RAM guards + disk spill backstop it. Tune via M3_APC_BLOCKS after measuring real bytes/block.
    _apc_blocks = int(os.environ.get("M3_APC_BLOCKS", "4096"))
    # [inc4b] APC disk persistence: when M3_APC_DISK points at a directory, back the manager with a
    # DiskBlockStore so hot prefixes (byron's system prompt, Alfred's history) survive a restart —
    # warm-restart prefix reuse instead of a cold prefill after every launchd relaunch. Verified
    # constructor: DiskBlockStore(root, namespace="default", num_workers=1, max_bytes=None) — root is
    # positional and internally wrapped in Path(), so a str dir path is accepted. Default: env unset ->
    # disk=None -> today's memory-only behavior, unchanged.
    _apc_disk = None
    _apc_disk_dir = os.environ.get("M3_APC_DISK")
    if _apc_disk_dir:
        _apc_disk = _apc_mod.DiskBlockStore(_apc_disk_dir)
        print("APC disk store enabled: %s (warm-restart prefix reuse)" % _apc_disk_dir, flush=True)
    APC = _apc_mod.APCManager(num_blocks=_apc_blocks, block_size=16, disk=_apc_disk)
    print("APC enabled: %d blocks x 16 tok (~%dK tok keyed prefix cache)" % (_apc_blocks, _apc_blocks * 16 // 1000), flush=True)
# The generation path caches when EITHER mechanism is on.
USE_CACHE = M3_APC or PREFIX_CACHE


class StructuredDecodingUnavailable(RuntimeError):
    pass


def _build_json_lp(response_format, think, explicit=None):
    """[increment-1 #1] Build a constrained-JSON logits processor from an OpenAI `response_format`.
    Supports {"type":"json_schema","json_schema":{"schema":{...}}} (grammar-enforced to the schema)
    and {"type":"json_object"} (enforced to be *some* valid JSON object). Returns a 1-element
    logits_processors list, or None only when response_format was not requested.
    The processor is ThinkingAware (M3's </mm:think>) so it only constrains the ANSWER, never the
    private chain-of-thought. Explicit requests fail closed: malformed formats raise ValueError
    (HTTP 400), while missing/broken structured runtime raises StructuredDecodingUnavailable (503).
    They are never silently enqueued as unconstrained generations."""
    if explicit is None:
        explicit = response_format is not None
    if not explicit:
        return None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    rtype = response_format.get("type")
    if rtype == "json_schema":
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            raise ValueError("response_format.json_schema must be an object")
        schema = json_schema.get("schema")
        if not isinstance(schema, dict) or not schema:
            raise ValueError("response_format.json_schema.schema must be a non-empty object")
    elif rtype == "json_object":
        schema = {"type": "object"}  # any valid JSON object
    else:
        raise ValueError("unsupported response_format type")
    if not _STRUCTURED_OK:
        _log_error("_build_json_lp", "structured decoding requested but unavailable")
        raise StructuredDecodingUnavailable("structured decoding unavailable")
    tok = getattr(PROC, "tokenizer", PROC)
    try:
        proc = build_json_schema_logits_processor(tok, schema)
    except (TypeError, ValueError) as e:
        _log_exc("_build_json_lp invalid schema")
        raise ValueError("invalid response_format schema") from e
    except Exception as e:
        _log_exc("_build_json_lp")
        raise StructuredDecodingUnavailable("structured decoding unavailable") from e
    try:
        proc = ThinkingAwareLogitsProcessor(proc, tok, thinking_end_token="</mm:think>",
                                            enable_thinking=(think == "enabled"))
        return [proc]
    except Exception as e:
        _log_exc("_build_json_lp")
        raise StructuredDecodingUnavailable("structured decoding unavailable") from e


def _gen(prompt, mt, temp, use_cache, images=None, cancel=None, logits_processors=None, stream_q=None):
    """One generation. use_cache=True reuses the shared prefix cache; False = fresh cache.
    logits_processors (increment-1 #1): when set (constrained JSON), force the STREAMED path so the
    processor is applied per token, and skip the cache (a grammar-constrained decode must not reuse
    another turn's KV blindly). The streamed path also gives us the wall-clock cap + cancel checks.
    images (non-empty list of file paths) -> vision turn: bypass the shared text-only prefix cache
    entirely (it would corrupt a vision turn) and run a fresh generate() with the image(s)).
    [auditfix] #3/#4: the streamed (use_cache) path enforces the MAX_GEN_S wall-clock cap and an
    optional cancel() callable (caller gave up) checked every 16 chunks; on abort the generator is
    close()d and mx.synchronize() drains the pending async_eval (GLM fix-5 lesson: a bare break
    leaves lazy graph + Metal buffers queued). The fresh-cache and vision generate() fallbacks are
    single blocking calls and cannot be capped mid-flight (documented limitation)."""
    if images:
        with contextlib.redirect_stdout(io.StringIO()):
            return generate(MODEL, PROC, prompt, image=images, max_tokens=mt,
                            temperature=temp, verbose=True)
    with contextlib.redirect_stdout(io.StringIO()):
        # [inc3] a streaming job (stream_q set) ALWAYS takes the streamed path so we can push tokens
        # incrementally, even when no cache mechanism is active.
        if use_cache or logits_processors or stream_q is not None:
            text = ""; last = None
            # constrained decode: skip cache (grammar state must not inherit another turn's KV).
            if logits_processors:
                _cache_kw = {"logits_processors": logits_processors}
            elif use_cache and APC is not None:  # APC (keyed, pollution-free) when enabled
                _cache_kw = {"apc_manager": APC}
            elif use_cache:  # legacy single shared PCSTATE (off-by-default; H3)
                _cache_kw = {"prompt_cache_state": PCSTATE}
            else:  # [inc3] streaming with no cache mechanism -> fresh decode (never re-arm PCSTATE)
                _cache_kw = {}
            try:
                gen = stream_generate(MODEL, PROC, prompt, max_tokens=mt,          # [auditfix] #3
                                      temperature=temp,
                                      prefill_step_size=PREFILL_STEP,               # [inc4a] OOM-guard chunked prefill
                                      **_cache_kw)
                t0 = time.time(); n = 0                                            # [auditfix] #3
                try:
                    for ch in gen:
                        text += ch.text; last = ch; n += 1
                        if stream_q is not None:                                  # [inc3] push raw token text
                            with contextlib.suppress(Exception):
                                stream_q.put(("tok", ch.text))
                        if CLEAR_CACHE_STEPS and (n % CLEAR_CACHE_STEPS) == 0:     # [auditfix] #2 (opt-in)
                            with contextlib.suppress(Exception):
                                mx.clear_cache()
                        if (n & 15) == 0:
                            if (time.time() - t0) > MAX_GEN_S:                     # [auditfix] #3 time-cap
                                print("[cap] GEN TIME-CAP %ds hit (%d chunks) — aborting to free the worker" % (MAX_GEN_S, n), flush=True)
                                with contextlib.suppress(Exception):
                                    last.finish_reason = "length"
                                break
                            if cancel is not None and cancel():                    # [auditfix] #4 dead caller
                                print("[cap] caller cancelled after %d chunks — aborting" % n, flush=True)
                                with contextlib.suppress(Exception):
                                    last.finish_reason = "length"
                                break
                finally:
                    with contextlib.suppress(Exception):
                        gen.close()        # [auditfix] #3: run the generator's finally (drop in-flight graph)
                    with contextlib.suppress(Exception):
                        mx.synchronize()   # [auditfix] #3: drain any pending async_eval
                if last is None:
                    if logits_processors:
                        # A structured request must never fall back to generate() without its
                        # per-token grammar processor.
                        raise RuntimeError("structured generation produced no streamed result")
                    fallback = generate(MODEL, PROC, prompt, max_tokens=mt,
                                        temperature=temp, verbose=True)
                    if stream_q is not None:
                        fallback_text = getattr(fallback, "text", "") or ""
                        if fallback_text:
                            stream_q.put(("tok", fallback_text))
                    return fallback
                last.text = text
                return last
            finally:
                # End must be queued after every streamed token, including the blocking empty-stream
                # fallback above; on exceptions it still terminates the SSE drain exactly once.
                if stream_q is not None:
                    with contextlib.suppress(Exception):
                        stream_q.put(("done", None))
        return generate(MODEL, PROC, prompt, max_tokens=mt, temperature=temp, verbose=True)


def _degenerate(out):
    """Detect output that must not be returned as-is. 'empty' = no usable content (the reliability
    blemish the gauntlet found, e.g. the reject_csp blank). 'loop' = runaway native tool-call token
    loop -- HIGH threshold so legit (few) tool calls are never flagged."""
    t = getattr(out, "text", "") or ""
    if not t.strip():
        return "empty"
    if t.count("<invoke name") > 12 or t.count("]<]minimax[>[") > 16:
        return "loop"
    return None


def _run(q, mt, temp, think, tools, images=None, cancel=None, logits_processors=None, stream_q=None):
    """Apply chat template + generate. Sheds load (raises Busy) when the GPU is busy.
    EMPTY/LOOP GUARD: if the first generation is degenerate, retry ONCE with a fresh cache and a
    temperature bump to break the greedy failure path, so a blank/looping reply never reaches the
    caller (Slack/trader). The watchdog remains the backstop for a hung server."""
    kw = {"thinking_mode": think}
    if tools:
        kw["tools"] = tools
    n_img = len(images) if images else 0
    if cancel is not None and cancel():  # [auditfix] #4: caller already gone — don't generate at all
        raise Busy()
    if not GEN_LOCK.acquire(timeout=LOCK_WAIT_S):
        raise Busy()
    try:
        prompt = apply_chat_template(PROC, CFG, q, num_images=n_img, **kw)
        out = _gen(prompt, mt, temp, USE_CACHE, images=images, cancel=cancel, logits_processors=logits_processors, stream_q=stream_q)  # [auditfix] #3/#4; [inc3] stream_q
        bad = _degenerate(out)
        # [inc3] NEVER retry a streaming job: its first-pass tokens were already pushed to the client
        # (and the end-of-stream sentinel already sent), so a second _gen would double-emit / stream to
        # a queue nobody drains. Streaming forgoes the empty/loop retry guard by design.
        if bad and mt > 0 and stream_q is None and not (cancel is not None and cancel()):  # [auditfix] #4: no retry for a dead caller; #6: mt<=0 is legitimately empty
            print("[guard] degenerate output (%s); retrying once (fresh cache, temp>=0.4)" % bad, flush=True)
            out = _gen(prompt, mt, max(temp, 0.4), False, images=images, cancel=cancel, logits_processors=logits_processors)
            if _degenerate(out):
                print("[guard] retry still degenerate; returning as-is", flush=True)
    finally:
        with contextlib.suppress(Exception):
            mx.clear_cache()  # [auditfix] #2: free the Metal buffer cache after every solo gen
        GEN_LOCK.release()
    return out


# [auditfix] #15 / [inc3] shared fallback for an unterminated think block (thinking explicitly ON but
# the </mm:think> close tag never arrived — token exhaustion mid-reasoning). Used by both the
# non-streaming strip_think and the SSE streaming path so the two never drift.
_THINK_FALLBACK = ("[no final answer: generation ended while still thinking (token limit hit "
                   "mid-reasoning). Retry with a higher max_tokens or thinking disabled.]")


def strip_think(text, think="disabled"):
    """Strip MiniMax-M3 chain-of-thought from the reply. The chat template opens the think block in
    the PROMPT, so generation is '<cot>...</mm:think><answer>' -- drop everything up to and including
    the close tag. Token usage/tps are left untouched (thinking genuinely generated those tokens).
    [auditfix] #15: when thinking was ENABLED and the close tag never arrived (token exhaustion
    mid-think), the WHOLE text is raw chain-of-thought — returning it as-is leaked CoT to the
    trader/Slack. Return a safe fallback instead. Must be think-mode-aware: thinking-OFF replies
    also have no close tag and MUST pass through unchanged (default keeps old behavior)."""
    if not text:
        return text
    m = re.search(r'</mm:think>', text)
    if m:
        return text[m.end():].lstrip()
    if think == "enabled":  # [auditfix] #15: unterminated think block -> never leak raw CoT
        return _THINK_FALLBACK
    return text


def parse_minimax_tools(text, tools):
    """Convert MiniMax-M3's native tool-call output into OpenAI `tool_calls`.
    M3 emits: <tool_call><invoke name="FN"><param>value</param>...</invoke></tool_call>,
    interleaved with the ']<]minimax[>[' delimiter token. hermes expects OpenAI tool_calls;
    without this the call leaks into `content` as raw text and never fires (the Slack leak).
    Returns (tool_calls|None, content). Always strips the stray delimiter token from content."""
    clean = (text or "").replace("]<]minimax[>[", "")
    if not tools:
        return None, (clean.strip() or text)
    invokes = re.findall(r'<invoke name="([^"]+)">(.*?)</invoke>', clean, re.S)
    if not invokes:
        return None, (clean.strip() or text)
    calls = []
    for i, (name, body) in enumerate(invokes):
        args = {}
        for pn, pv in re.findall(r'<([A-Za-z0-9_.\-]+)>(.*?)</\1>', body, re.S):
            v = pv.strip()
            try:
                args[pn] = json.loads(v)
            except Exception:
                args[pn] = v
        calls.append({"id": "call_%d" % i, "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}})
    cut = re.search(r'<tool_call>|<invoke name=', clean)
    pre = clean[:cut.start()].strip() if cut else ""
    return calls, (pre or None)


M3_MAX_IMG_BYTES = int(os.environ.get("M3_MAX_IMG_BYTES", str(12 * 1024 * 1024)))
# The 20 MiB default accommodates one 12 MiB decoded image (~16 MiB base64) plus JSON/message
# overhead while preventing an unauthenticated local/proxied caller from making an unbounded read.
M3_MAX_BODY_BYTES = int(os.environ.get("M3_MAX_BODY_BYTES", str(20 * 1024 * 1024)))
if M3_MAX_BODY_BYTES < 1024:
    raise ValueError("M3_MAX_BODY_BYTES must be at least 1024")
M3_MAX_TOKENS = int(os.environ.get("M3_MAX_TOKENS", "16384"))
M3_MAX_TEMPERATURE = float(os.environ.get("M3_MAX_TEMPERATURE", "2.0"))
if M3_MAX_TOKENS < 1 or not math.isfinite(M3_MAX_TEMPERATURE) or M3_MAX_TEMPERATURE < 0:
    raise ValueError("invalid M3_MAX_TOKENS / M3_MAX_TEMPERATURE configuration")


def _extract_images(msgs):
    """Pull OpenAI multimodal image_url parts out of message content and write them to temp files.
    For each message whose content is a LIST, base64-decode any image_url data URLs to temp .img files,
    collect their paths, and flatten that message content to the joined text of its text parts.
    If content is NOT a list (a plain string), it is left untouched -- the text/trading path is unchanged.
    Returns (msgs, image_paths, tmp_paths). msgs is a NEW list (originals not mutated)."""
    image_paths = []
    tmp_paths = []
    out_msgs = []
    for m in msgs:
        c = m.get("content")
        if not isinstance(c, list):
            out_msgs.append(m)
            continue
        texts = []
        for part in c:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(part.get("text") or "")
            elif part.get("type") == "image_url":
                url = ((part.get("image_url") or {}).get("url")) or ""
                if url.startswith("data:") and "base64," in url:
                    try:  # [auditfix] #8: malformed base64 must not leak temps already written
                        b64 = url.split("base64,", 1)[1]
                        raw = base64.b64decode(b64, validate=True)
                    except Exception:
                        for tp in tmp_paths:
                            with contextlib.suppress(Exception):
                                os.unlink(tp)
                        raise ValueError("malformed base64 in image_url data URL")
                    fd, path = tempfile.mkstemp(suffix=".img")
                    with os.fdopen(fd, "wb") as f:
                        f.write(raw)
                    image_paths.append(path)
                    tmp_paths.append(path)
        nm = dict(m)
        nm["content"] = "\n".join(t for t in texts if t)  # [auditfix] #11: was "\\n" -> a visible literal backslash-n in the prompt
        out_msgs.append(nm)
    return out_msgs, image_paths, tmp_paths


# `save_to` is disabled unless the operator explicitly opts in. When enabled, its directory must
# be owned by the service user with no group/other permissions, and writes are direct-child,
# no-symlink, create-only operations. Existing files can never be overwritten.
ALLOW_SAVE_TO = os.environ.get("M3_ALLOW_SAVE_TO", "0") == "1"
SAVE_DIR = os.path.realpath(os.environ.get("M3_SAVE_DIR", os.path.expanduser("~/m3_saves")))


def _save_text_create_only(p, text):
    """Write *text* as a new owner-only direct child of SAVE_DIR.

    Returns (True, None) on success or (False, public_error) on rejection/failure. Internal
    exception details go only to stderr via _log_exc.
    """
    if not ALLOW_SAVE_TO:
        return False, "save_to disabled"
    try:
        save_root = os.path.realpath(SAVE_DIR)
        os.makedirs(save_root, mode=0o700, exist_ok=True)
        dst = os.path.abspath(str(p)) if os.path.isabs(str(p)) else os.path.join(save_root, str(p))
        name = os.path.basename(dst)
        if not name or name in (".", "..") or os.path.realpath(os.path.dirname(dst)) != save_root:
            return False, "save_to rejected"

        st = os.stat(save_root, follow_symlinks=False)
        if (not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid()
                or stat.S_IMODE(st.st_mode) & 0o077):
            return False, "save_to directory must be owner-only"

        dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        dfd = os.open(save_root, dir_flags)
        try:
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(name, file_flags, 0o600, dir_fd=dfd)
        finally:
            os.close(dfd)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        return True, None
    except FileExistsError:
        return False, "save_to target already exists"
    except Exception:
        _log_exc("_save_text_create_only")
        return False, "save_to unavailable"



# ===== BATCHING WORKER (added for m3_serve_batched) =====================================
import queue as _queue
try:
    from m3_batch_core import generate_batch
except Exception as _e:
    generate_batch = None
    if _RUNTIME_IMPORT_ERROR is None:
        _RUNTIME_IMPORT_ERROR = _e
BATCH_MAX = int(os.environ.get("M3_BATCH_MAX", "4"))
# [auditfix] concurrency: 12 -> 150ms. A 12ms window almost never actually coalesced real traffic
# (Slack-Alfred's parallel calls arrive over tens of ms; trader+chat overlap over seconds), so in
# practice everything ran solo/serially. 150ms is still invisible next to multi-second gens but
# catches genuinely-concurrent arrivals into one forward.
BATCH_WINDOW_MS = int(os.environ.get("M3_BATCH_WINDOW_MS", "150"))
# [auditfix] concurrency: 2000 -> 6144. Unlike GLM (whose <2048 gate is load-bearing for the DSA
# indexer), M3's Lightning Indexer is INERT at serve time (dense attention everywhere), and the
# batch core's left-pad masking is correct at ANY length — the only real gate is KV memory.
# Worst case 4 rows x 6144 tok ≈ 24.5k KV positions, a few GB against this box's headroom; prompts
# longer than the gate still run solo. Raise/lower via env after live validation.
BATCH_MAX_PROMPT_TOK = int(os.environ.get("M3_BATCH_MAX_PROMPT_TOK", "6144"))  # prompts longer -> run solo, never batched
# [inc2] PRIORITY SCHEDULING: JOBQ is now a PriorityQueue of (priority, seq, job) tuples ordered by
# (priority, seq). priority 0=urgent/trader, 1=default, 2=bench (lower number = served first). `seq` is
# a strictly-increasing tiebreaker so (a) same-priority jobs stay FIFO and (b) the tuple ordering never
# has to compare two _Job objects (they are not comparable). Env-less behavior is unchanged: absent the
# X-M3-Priority header every request is priority 1, so a single-class stream behaves exactly like today.
JOBQ = _queue.PriorityQueue()
_SEQ_LOCK = threading.Lock()
_SEQ = 0
def _next_seq():
    global _SEQ
    with _SEQ_LOCK:
        _SEQ += 1
        return _SEQ
MAX_PENDING = int(os.environ.get("M3_MAX_PENDING", "8"))  # shed (503) when pending backlog >= this
EOS_IDS = set()

# --- metrics (increment-1 #5): dependency-free Prometheus text exposition on GET /metrics.
# Feeds the existing LGTM stack on k3s (queue depth, gen latency/throughput, 503 sheds, batch
# occupancy, constrained-JSON usage, and — most valuable — a Metal-memory gauge for OOM early
# warning). Counters are plain ints under a lock; no external dependency.
_METRICS = {"requests_total": 0, "busy_503_total": 0, "errors_total": 0, "gen_count": 0,
            "gen_seconds_sum": 0.0, "tokens_out_total": 0, "prompt_tokens_total": 0,
            "batched_gens_total": 0, "solo_gens_total": 0, "constrained_total": 0,
            "batch_rows_sum": 0, "batch_count": 0,
            "p0_total": 0, "p1_total": 0, "p2_total": 0}  # [inc2] per-priority-class request counters
_METLOCK = threading.Lock()
def _metric_add(**kw):
    with _METLOCK:
        for k, v in kw.items():
            _METRICS[k] = _METRICS.get(k, 0) + v
def _metrics_text():
    with _METLOCK:
        m = dict(_METRICS)
    try:
        active = int(mx.get_active_memory()); peak = int(mx.get_peak_memory())
    except Exception:
        active = peak = 0
    avg_batch = (m["batch_rows_sum"] / m["batch_count"]) if m["batch_count"] else 0.0
    lines = [
        "# HELP m3_requests_total OpenAI+legacy requests received", "# TYPE m3_requests_total counter",
        "m3_requests_total %d" % m["requests_total"],
        "# TYPE m3_busy_503_total counter", "m3_busy_503_total %d" % m["busy_503_total"],
        "# TYPE m3_errors_total counter", "m3_errors_total %d" % m["errors_total"],
        "# TYPE m3_gen_seconds_sum counter", "m3_gen_seconds_sum %.3f" % m["gen_seconds_sum"],
        "# TYPE m3_gen_count counter", "m3_gen_count %d" % m["gen_count"],
        "# TYPE m3_tokens_out_total counter", "m3_tokens_out_total %d" % m["tokens_out_total"],
        "# TYPE m3_prompt_tokens_total counter", "m3_prompt_tokens_total %d" % m["prompt_tokens_total"],
        "# TYPE m3_solo_gens_total counter", "m3_solo_gens_total %d" % m["solo_gens_total"],
        "# TYPE m3_batched_gens_total counter", "m3_batched_gens_total %d" % m["batched_gens_total"],
        "# TYPE m3_constrained_total counter", "m3_constrained_total %d" % m["constrained_total"],
        "# HELP m3_p0_total priority-0 (urgent/trader) requests", "# TYPE m3_p0_total counter",
        "m3_p0_total %d" % m["p0_total"],
        "# HELP m3_p1_total priority-1 (default) requests", "# TYPE m3_p1_total counter",
        "m3_p1_total %d" % m["p1_total"],
        "# HELP m3_p2_total priority-2 (bench) requests", "# TYPE m3_p2_total counter",
        "m3_p2_total %d" % m["p2_total"],
        "# HELP m3_queue_depth pending jobs in the worker queue", "# TYPE m3_queue_depth gauge",
        "m3_queue_depth %d" % JOBQ.qsize(),
        "# TYPE m3_avg_batch_rows gauge", "m3_avg_batch_rows %.2f" % avg_batch,
        "# HELP m3_gpu_active_bytes MLX active Metal memory", "# TYPE m3_gpu_active_bytes gauge",
        "m3_gpu_active_bytes %d" % active,
        "# TYPE m3_gpu_peak_bytes gauge", "m3_gpu_peak_bytes %d" % peak,
        "# TYPE m3_ready gauge", "m3_ready %d" % (1 if READY else 0),
    ]
    return ("\n".join(lines) + "\n").encode()

class _Job:
    __slots__ = ("msgs", "mt", "temp", "think", "tools", "images", "lp", "ev", "result", "error",
                 "cancelled", "priority", "stream_q")  # [auditfix] #4; +lp (#1); +priority (inc2); +stream_q (inc3)
    def __init__(self, msgs, mt, temp, think, tools, images, lp=None, priority=1, stream_q=None):
        self.msgs, self.mt, self.temp, self.think, self.tools, self.images = msgs, mt, temp, think, tools, images
        self.lp = lp  # logits_processors (constrained JSON) -> forces solo, threads to _run
        self.priority = priority  # [inc2] 0=urgent/trader, 1=default, 2=bench
        self.stream_q = stream_q  # [inc3] queue.Queue for SSE token streaming -> forces solo, threads to _run
        self.ev = threading.Event(); self.result = None; self.error = None
        self.cancelled = False  # [auditfix] #4: set when the caller times out/hangs up; worker skips/aborts the job

class _GenOut:
    def __init__(self, text, pt, ct, fr):
        self.text, self.prompt_tokens, self.generation_tokens, self.finish_reason = text, pt, ct, fr
        self.logprobs = None

def _enqueue(msgs, mt, temp, think, tools, images=None, lp=None, priority=1, stream_q=None):
    """[inc2/inc3] Shed by priority, build a _Job, and put (priority, seq, job) on the PriorityQueue.
    Returns the enqueued job WITHOUT waiting (the streaming path drains stream_q itself; the blocking
    path waits on j.ev via submit_and_wait). Raises Busy when the backlog shed rejects the request.
    Shed policy (MAX_PENDING = default backlog bound):
      * priority 0 (urgent/trader): EXEMPT — always queued, even at backlog (never dropped).
      * priority 2 (bench): sheds FIRST — threshold MAX_PENDING//2 so bench yields headroom early.
      * priority 1 (default): unchanged — sheds at MAX_PENDING."""
    depth = JOBQ.qsize()
    if priority == 2:
        if depth >= max(1, MAX_PENDING // 2):
            raise Busy()
    elif priority != 0:  # default class (1); priority 0 is exempt from the shed entirely
        if depth >= MAX_PENDING:
            raise Busy()
    _metric_add(**{"p%d_total" % priority: 1})  # [inc2] per-priority-class counter (admitted requests)
    j = _Job(msgs, mt, temp, think, tools, images, lp=lp, priority=priority, stream_q=stream_q)
    JOBQ.put((priority, _next_seq(), j))
    return j

def submit_and_wait(msgs, mt, temp, think, tools, images=None, lp=None, priority=1):
    j = _enqueue(msgs, mt, temp, think, tools, images=images, lp=lp, priority=priority)
    if not j.ev.wait(timeout=LOCK_WAIT_S):
        j.cancelled = True  # [auditfix] #4: caller is giving up — worker must not (keep) generating for a dead caller
        raise Busy()
    if j.error: raise j.error
    return j.result

def _tok_ids(j):
    kw = {"thinking_mode": j.think}
    if j.tools: kw["tools"] = j.tools
    prompt = apply_chat_template(PROC, CFG, j.msgs, num_images=0, **kw)
    tk = getattr(PROC, "tokenizer", PROC)
    return tk.encode(prompt) if isinstance(prompt, str) else prompt

def _do_solo(j):
    if j.cancelled:  # [auditfix] #4: caller already timed out — don't generate for a dead caller
        j.ev.set(); return
    _t0 = time.time()
    try:
        j.result = _run(j.msgs, j.mt, j.temp, j.think, j.tools, images=j.images, cancel=lambda: j.cancelled, logits_processors=j.lp, stream_q=j.stream_q)  # [auditfix] #4; #1 constrained; [inc3] stream_q
        _metric_add(solo_gens_total=1, gen_count=1, gen_seconds_sum=time.time() - _t0,
                    tokens_out_total=int(getattr(j.result, "generation_tokens", 0) or 0),
                    prompt_tokens_total=int(getattr(j.result, "prompt_tokens", 0) or 0))
    except Exception as e:
        j.error = e; _metric_add(errors_total=1)
        _log_exc("_do_solo")  # [auditfix] #5
    finally: j.ev.set()

def _do_batched(jobs, id_lists):
    tk = getattr(PROC, "tokenizer", PROC)
    mts = [j.mt for j in jobs]
    # [auditfix] #1: take GEN_LOCK around the batched forward. Previously generate_batch ran with
    # NO lock while _legacy called _run() directly on the handler thread -> two simultaneous MLX
    # generations on one Metal queue (corruption + doubled peak memory). _legacy is now also routed
    # through the worker queue (see _legacy), so this lock is uncontended defense-in-depth.
    if not GEN_LOCK.acquire(timeout=LOCK_WAIT_S):
        raise Busy()
    try:
        seed = random.getrandbits(31)  # [auditfix] #10: was the implicit seed=0 EVERY batch -> identical "random" outputs forever at temp>0
        outs = generate_batch(MODEL, id_lists, mts, EOS_IDS, temp=jobs[0].temp, seed=seed,
                              cancels=[(lambda j=j: j.cancelled) for j in jobs],  # [auditfix] #4: rows drop mid-decode when their caller dies
                              max_gen_s=MAX_GEN_S,                                # [auditfix] #3: wall-clock cap on the batched decode loop
                              clear_cache_steps=CLEAR_CACHE_STEPS)                # [auditfix] #2 (opt-in periodic trim)
    finally:
        with contextlib.suppress(Exception):
            mx.clear_cache()  # [auditfix] #2: free the Metal buffer cache after every batched gen
        GEN_LOCK.release()
    _metric_add(batched_gens_total=1, batch_count=1, batch_rows_sum=len(jobs))
    for j, ids, oids in zip(jobs, id_lists, outs):
        fr = "length"
        if oids and oids[-1] in EOS_IDS:
            fr = "stop"; oids = oids[:-1]
        res = _GenOut(tk.decode(oids), len(ids), len(oids), fr)
        if j.cancelled:  # [auditfix] #4: caller timed out mid-batch — nothing to deliver
            j.ev.set(); continue
        if j.mt > 0 and _degenerate(res):  # [auditfix] #6: blank/loop guard now covers batched rows; requeue bad rows solo
            print("[guard] batched row degenerate (%s); re-running solo with retry guard" % _degenerate(res), flush=True)
            _do_solo(j)  # sets j.ev itself; solo path has the fresh-cache + temp-bump retry
        else:
            j.result = res; j.ev.set()

def _batch_worker():
    while True:
        first = JOBQ.get()[2]   # [inc2] PriorityQueue yields (priority, seq, job); unwrap the job
        jobs = [first]
        p0 = first.priority
        try:  # [auditfix] #16: see except at the bottom — a worker escape must not be a silent brownout
            # [inc2] priority-0 (urgent/trader) SKIPS the 150ms coalescing window and runs immediately
            # with whatever is already in hand (a batch of 1). For any other class the window gathers
            # ONLY jobs of the SAME priority as the first job: a job pulled from the queue whose class
            # differs (e.g. a bench job, or a just-arrived urgent one) is put straight back and ends the
            # window, so no bench job ever rides a trader/default batch.
            if p0 != 0:
                t0 = time.time()
                while len(jobs) < BATCH_MAX:
                    rem = BATCH_WINDOW_MS / 1000.0 - (time.time() - t0)
                    if rem <= 0: break
                    try: nitem = JOBQ.get(timeout=rem)
                    except _queue.Empty: break
                    if nitem[2].priority != p0:
                        JOBQ.put(nitem)  # different class -> leave it for the next worker loop
                        break
                    jobs.append(nitem[2])
            live = []
            for j in jobs:  # [auditfix] #4: drop jobs whose caller already timed out while queued
                if j.cancelled: j.ev.set()
                else: live.append(j)
            jobs = live
            if not jobs:
                continue
            # vision never batches; constrained-JSON (j.lp) never batches (batch core has no
            # logits-processor hook — the trader's structured requests run solo, which is fine:
            # the trader fires one at a time and byron is single-threaded). [inc3] streaming jobs
            # (j.stream_q) also never batch — token streaming needs the solo stream_generate loop.
            for vj in (j for j in jobs if j.images or j.lp or j.stream_q is not None): _do_solo(vj)
            # length-gate: long prompts run SOLO (one big KV at a time); short ones batch
            short, short_ids = [], []
            for j in (j for j in jobs if not j.images and not j.lp and j.stream_q is None):
                try:
                    ids = _tok_ids(j)
                except Exception as e:
                    j.error = e; j.ev.set(); _log_exc("_tok_ids"); continue  # [auditfix] #5
                if len(ids) > BATCH_MAX_PROMPT_TOK:
                    _do_solo(j)
                else:
                    short.append(j); short_ids.append(ids)
            if len(short) == 1:
                _do_solo(short[0])
            elif len(short) > 1:
                if len({round(j.temp, 4) for j in short}) == 1:
                    try: _do_batched(short, short_ids)
                    except Exception as e:
                        _log_exc("_do_batched")  # [auditfix] #5: batch failures now hit err.log
                        for j in short: j.error = e; j.ev.set()
                else:
                    for j in short: _do_solo(j)
        except BaseException:
            # [auditfix] #16: worker death used to be a SILENT BROWNOUT — the thread died, /health
            # kept reporting ready:true, and every POST queued until its 600s timeout. Mirror the
            # GLM H4 approach: log, fail the in-hand jobs, and hard-exit non-zero so launchd
            # KeepAlive relaunches a healthy process.
            _log_exc("_batch_worker FATAL — exiting for KeepAlive relaunch")
            err = RuntimeError("batch worker died; server restarting")
            for j in jobs:
                with contextlib.suppress(Exception):
                    if not j.ev.is_set():
                        j.error = err; j.ev.set()
            os._exit(1)
# ===== END BATCHING WORKER =============================================================


class RequestBodyError(Exception):
    def __init__(self, status, message, error_type="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type


def _request_error_body(err):
    return json.dumps({"error": {"message": err.message, "type": err.error_type}}).encode()


def _parse_content_length(headers, max_bytes=None):
    """Return one canonical non-negative Content-Length or raise RequestBodyError."""
    limit = M3_MAX_BODY_BYTES if max_bytes is None else int(max_bytes)
    transfer_encoding = headers.get("Transfer-Encoding")
    if transfer_encoding:
        raise RequestBodyError(400, "Transfer-Encoding is not supported")
    values = headers.get_all("Content-Length", []) if hasattr(headers, "get_all") else []
    if not values:
        raw = headers.get("Content-Length")
        values = [] if raw is None else [raw]
    if not values:
        raise RequestBodyError(411, "Content-Length is required")
    if len(values) != 1:
        raise RequestBodyError(400, "exactly one Content-Length header is required")
    raw = str(values[0]).strip()
    if not raw:
        raise RequestBodyError(411, "Content-Length is required")
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise RequestBodyError(400, "invalid Content-Length")
    canonical = raw.lstrip("0") or "0"
    lim = str(limit)
    if len(canonical) > len(lim) or (len(canonical) == len(lim) and canonical > lim):
        raise RequestBodyError(413, "request body too large", "payload_too_large")
    return int(canonical)


def _read_json_request(headers, rfile, max_bytes=None):
    n = _parse_content_length(headers, max_bytes=max_bytes)
    body = rfile.read(n)
    if len(body) != n:
        raise RequestBodyError(400, "incomplete request body")
    try:
        req = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RequestBodyError(400, "request body must be valid JSON")
    if not isinstance(req, dict):
        raise RequestBodyError(400, "request body must be a JSON object")
    return req


PRIORITY_TOKEN_FILE = os.environ.get("M3_PRIORITY_TOKEN_FILE", "")


def _load_priority_token(path=None):
    """Read a token only from a regular, owner-owned file inaccessible to group/other users."""
    token_path = PRIORITY_TOKEN_FILE if path is None else path
    if not token_path:
        return None
    fd = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(token_path, flags)
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid()
                or stat.S_IMODE(st.st_mode) & 0o077 or st.st_size > 4096):
            return None
        raw = os.read(fd, 4097)
        token = raw.decode("utf-8").strip()
        return token if 32 <= len(token) <= 4096 else None
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if fd is not None:
            os.close(fd)


def _request_priority(headers):
    """Honor bench/yield priority 2 freely; require the owner-file credential for urgent 0."""
    try:
        requested = int(headers.get("X-M3-Priority", "1"))
    except (TypeError, ValueError):
        return 1
    if requested == 2:  # safe downgrade: preserves Byron/gauntlet yielding behavior
        return 2
    if requested != 0:
        return 1
    expected = _load_priority_token()
    supplied = headers.get("X-M3-Priority-Token", "")
    if expected and isinstance(supplied, str) and hmac.compare_digest(supplied, expected):
        return requested
    return 1


def _validate_openai_fields(req):
    """Validate fields consumed before generation and return (messages, max_tokens, temperature)."""
    msgs = req.get("messages")
    if not isinstance(msgs, list) or any(not isinstance(m, dict) for m in msgs):
        raise ValueError("messages must be a list of objects")

    mt = req.get("max_tokens")
    if mt is None:
        mt = req.get("max_completion_tokens")
    if mt is None:
        mt = 4096
    if isinstance(mt, bool):
        raise ValueError("max_tokens must be an integer")
    if isinstance(mt, int):
        pass
    elif isinstance(mt, float) and math.isfinite(mt) and mt.is_integer():
        mt = int(mt)
    else:
        raise ValueError("max_tokens must be an integer")
    if mt < 0 or mt > M3_MAX_TOKENS:
        raise ValueError("max_tokens out of range")

    temp = req.get("temperature", 0.0)
    if isinstance(temp, bool) or not isinstance(temp, (int, float)):
        raise ValueError("temperature must be finite and numeric")
    if temp < 0 or temp > M3_MAX_TEMPERATURE:
        raise ValueError("temperature out of range")
    temp = float(temp)
    if not math.isfinite(temp):
        raise ValueError("temperature must be finite and numeric")
    return msgs, mt, temp

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        """Send a response; swallow BrokenPipe/ConnectionReset if the caller already hung up."""
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

    def do_GET(self):
        # GET endpoints never touch the model -> always answerable, even mid-load. /v1/models and
        # /health are the watchdog's "is the server process alive" liveness checks.
        if self.path.startswith("/metrics"):   # #5: Prometheus text exposition
            self._send(200, _metrics_text(), ctype="text/plain; version=0.0.4")
        elif self.path.startswith("/v1/models") or self.path.startswith("/health"):
            body = json.dumps({"object": "list", "ready": READY,
                               "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}).encode()
            self._send(200, body)
        else:
            self._send(200, b"ready" if READY else b"loading", ctype="text/plain")

    def do_POST(self):
        route = self.path.partition("?")[0]
        if route not in ("/", "/v1/chat/completions"):
            self._send(404, json.dumps({"error": {"message": "route not found",
                                                   "type": "not_found_error"}}).encode())
            return
        # READY gate: real inference is held off until the weights finish loading. Return 503
        # "loading" so callers retry/fall back rather than hit an unloaded model. The listening
        # socket is already owned (bound before load) so this responds in milliseconds.
        if not READY:
            self._send(503, b'{"error":"loading"}')
            return
        try:
            req = _read_json_request(self.headers, self.rfile)
        except RequestBodyError as e:
            # We intentionally do not consume an oversized body; close the connection so its bytes
            # cannot be interpreted as a subsequent request on a persistent connection.
            self.close_connection = True
            self._send(e.status, _request_error_body(e))
            return
        prio = _request_priority(self.headers)
        try:
            if route == "/v1/chat/completions":
                return self._openai(req, prio)
            return self._legacy(req, prio)
        except StructuredDecodingUnavailable:
            self._send(503, json.dumps({"error": {"message": "structured decoding unavailable",
                                                   "type": "service_unavailable"}}).encode())
        except (TypeError, ValueError):
            self._send(400, json.dumps({"error": {"message": "invalid request parameters",
                                                   "type": "invalid_request_error"}}).encode())
        except Exception:
            _log_exc("do_POST")
            self._send(500, json.dumps({"error": {"message": "internal server error",
                                                   "type": "server_error"}}).encode())

    # ---- OpenAI-compatible chat completions (PROD) ----
    def _openai(self, req, prio=1):
        msgs, mt, temp = _validate_openai_fields(req)
        think = "disabled"  # prod default
        # honor a leading "think:" on the last user message (preserves the Slack-Alfred prefix UX)
        if msgs and isinstance(msgs[-1].get("content"), str) and msgs[-1]["content"].lstrip().lower().startswith("think:"):
            think = "enabled"; msgs[-1]["content"] = msgs[-1]["content"].lstrip()[6:].lstrip()
        if req.get("thinking") in ("enabled", "disabled", "adaptive"):
            think = req["thinking"]
        tools = req.get("tools")
        lp = _build_json_lp(req.get("response_format"), think,
                            explicit=("response_format" in req))  # #1 fail-closed constrained JSON
        _metric_add(requests_total=1, **({"constrained_total": 1} if lp else {}))  # #5
        # Vision: pull any image_url parts out to temp files (text-only messages pass through untouched)
        try:  # [auditfix] #8: was OUTSIDE any try -> malformed base64 killed the handler with no HTTP reply
            msgs, image_paths, tmp_paths = _extract_images(msgs)
        except Exception:
            _log_exc("_extract_images")
            self._send(400, json.dumps({"error": {"message": "invalid image payload (malformed base64 / image_url part)",
                                                  "type": "invalid_request_error"}}).encode())
            return
        if image_paths:
            total = sum(os.path.getsize(p) for p in image_paths)
            if total > M3_MAX_IMG_BYTES:
                for tp in tmp_paths:
                    with contextlib.suppress(Exception): os.unlink(tp)
                err = json.dumps({"error": {"message": "image(s) too large (%d bytes > %d)" % (total, M3_MAX_IMG_BYTES), "type": "payload_too_large"}}).encode()
                self.send_response(413); self.send_header("Content-Type", "application/json")
                self.end_headers()
                with contextlib.suppress(Exception): self.wfile.write(err)
                return
        # [inc3] TRUE SSE token streaming (solo path): stream:true with NO tools streams tokens as they
        # decode (forces the job solo). Tool-call requests keep the whole-completion protocol-parity SSE
        # path below (partial tool calls are not streamed). _stream_tokens owns the tmp_paths cleanup.
        if req.get("stream") and not tools:
            return self._stream_tokens(req, msgs, mt, temp, think, image_paths, tmp_paths, lp, prio)
        try:
            out = submit_and_wait(msgs, mt, temp, think, tools, images=image_paths, lp=lp, priority=prio)
        except Busy:
            _metric_add(busy_503_total=1)  # #5
            err = json.dumps({"error": {"message": "model busy", "type": "overloaded"}}).encode()
            self.send_response(503); self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "2"); self.end_headers()
            with contextlib.suppress(Exception): self.wfile.write(err)
            return
        except Exception:
            _log_exc("_openai")  # [auditfix] #5: the 500 cause now reaches err.log
            err = json.dumps({"error": {"message": "internal server error", "type": "server_error"}}).encode()
            self.send_response(500); self.send_header("Content-Type", "application/json")
            self.end_headers()
            with contextlib.suppress(Exception): self.wfile.write(err)
            return
        finally:
            for tp in tmp_paths:
                with contextlib.suppress(Exception): os.unlink(tp)
        text = getattr(out, "text", "") or ""
        text = strip_think(text, think)  # [auditfix] #15: think-aware (unterminated CoT never leaks)
        pt = getattr(out, "prompt_tokens", None) or 0
        ct = getattr(out, "generation_tokens", None) or 0
        fr = getattr(out, "finish_reason", None) or "stop"
        tcalls, content = parse_minimax_tools(text, tools)
        msg = {"role": "assistant", "content": content}
        if tcalls:
            msg["tool_calls"] = tcalls
            fr = "tool_calls"
        if req.get("stream"):  # [auditfix] #9: stream:true was advertised but silently ignored
            return self._send_sse_completion(req, msg, fr, pt, ct)
        _choice = {"index": 0, "message": msg, "finish_reason": fr}
        if req.get("logprobs"):
            try:
                import mlx.core as _mx
                _lp = getattr(out, "logprobs", None)
                if _lp is not None:
                    _a = (_lp.astype(_mx.float32) if isinstance(_lp, _mx.array) else _mx.array(_lp, dtype=_mx.float32)).reshape(-1)
                    _k = int(req.get("top_logprobs") or 10)
                    _idx = _mx.argsort(_a)[-_k:].tolist()[::-1]
                    _tk = getattr(PROC, "tokenizer", None) or PROC
                    def _dec(_i):
                        try:
                            return _tk.decode([int(_i)])
                        except Exception:
                            try:
                                return _tk.convert_ids_to_tokens(int(_i))
                            except Exception:
                                return str(int(_i))
                    _choice["logprobs"] = {"content": [{"top_logprobs":
                        [{"token": _dec(_i), "logprob": float(_a[int(_i)].item())} for _i in _idx]}]}
            except Exception:
                _log_exc("_openai logprobs")
                _choice["logprobs"] = {"error": "logprobs unavailable"}
        resp = {"id": "chatcmpl-m3", "object": "chat.completion", "created": int(time.time()),
                "model": req.get("model") or MODEL_ID,
                "choices": [_choice],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}
        body = json.dumps(resp).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers()
        with contextlib.suppress(Exception): self.wfile.write(body)

    def _send_sse_completion(self, req, msg, fr, pt, ct):
        # [auditfix] #9: minimal OpenAI SSE. `stream:true` was advertised in the module docstring
        # but silently ignored — SSE clients waiting on an event stream got one plain JSON body.
        # The batched worker returns WHOLE completions, so this emits the finished result as a
        # spec-shaped chunk stream (role+content delta -> finish chunk -> [DONE]): protocol
        # parity, NOT incremental-latency streaming. True token streaming is deferred to the
        # mlx_lm.server continuous-batching cutover (see SERVING_FIXES_README).
        base = {"id": "chatcmpl-m3", "object": "chat.completion.chunk", "created": int(time.time()),
                "model": req.get("model") or MODEL_ID}
        d1 = {"role": "assistant"}
        if msg.get("content") is not None:
            d1["content"] = msg["content"]
        if msg.get("tool_calls"):
            d1["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(msg["tool_calls"])]
        chunks = [dict(base, choices=[{"index": 0, "delta": d1, "finish_reason": None}]),
                  dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": fr}],
                       usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})]
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.end_headers()
            for c in chunks:
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    def _stream_tokens(self, req, msgs, mt, temp, think, image_paths, tmp_paths, lp, prio):
        # [inc3] TRUE token streaming on the solo path. Enqueue a SOLO job carrying a stream_q, then
        # drain that queue and write spec-shaped OpenAI chat.completion.chunk SSE deltas as tokens land
        # (vs _send_sse_completion, which replays a finished completion for the batched/tool cases).
        # Think-handling: buffer output until </mm:think> is seen (emit only what follows) or — for a
        # NON-thinking request — 48 chars accumulate without it (emit everything buffered, then stream
        # freely). DEVIATION (documented): the 48-char free-stream escape is gated to think != "enabled".
        # When thinking is explicitly ON the model's output BEGINS inside the CoT (the template opened
        # the block), so applying the 48-char escape would leak raw reasoning; instead we keep buffering
        # until the close tag and, if it never arrives, emit the safe fallback — preserving the "never
        # leak unterminated CoT" invariant (strip_think #15). Early-cancel: a BrokenPipe/ConnectionReset
        # on any write sets j.cancelled (the decode loop aborts within 16 chunks) and stops the drain.
        sq = _queue.Queue()
        try:
            j = _enqueue(msgs, mt, temp, think, None, images=image_paths, lp=lp, priority=prio, stream_q=sq)
        except Busy:
            _metric_add(busy_503_total=1)
            for tp in tmp_paths:
                with contextlib.suppress(Exception): os.unlink(tp)
            err = json.dumps({"error": {"message": "model busy", "type": "overloaded"}}).encode()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.send_response(503); self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", "2"); self.end_headers(); self.wfile.write(err)
            return
        base = {"id": "chatcmpl-m3", "object": "chat.completion.chunk", "created": int(time.time()),
                "model": req.get("model") or MODEL_ID}
        state = {"role_sent": False, "broken": False}
        def _emit(text):
            if not text or state["broken"]:
                return
            delta = {"content": text}
            if not state["role_sent"]:
                delta = {"role": "assistant", "content": text}
                state["role_sent"] = True
            try:
                self.wfile.write(b"data: " + json.dumps(
                    dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}])).encode() + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                state["broken"] = True
                j.cancelled = True  # [inc3] early-cancel: abort the decode for a gone client
        try:
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            state["broken"] = True; j.cancelled = True
        try:
            buf = ""; streaming_free = False
            deadline = time.monotonic() + LOCK_WAIT_S
            while not state["broken"]:
                if j.ev.is_set() and sq.empty():
                    break  # pre-stream failure (e.g. template error) produced no sentinel
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    kind, payload = sq.get(timeout=min(0.25, remaining))
                except _queue.Empty:
                    continue
                if kind == "done":
                    break
                if streaming_free:
                    _emit(payload); continue
                buf += payload
                idx = buf.find("</mm:think>")
                if idx != -1:  # answer starts after the close tag; drop the CoT prefix
                    rest = buf[idx + len("</mm:think>"):].lstrip()
                    buf = ""; streaming_free = True
                    _emit(rest)
                elif think != "enabled" and len(buf) >= 48:  # no think block -> emit all, stream freely
                    _emit(buf); buf = ""; streaming_free = True
                # else: keep buffering (thinking request, or <48 chars and close tag not yet seen)
            completed = False
            if not state["broken"]:
                completed = j.ev.wait(timeout=LOCK_WAIT_S)
                if not completed:
                    j.cancelled = True
                if j.error is not None or not completed:
                    # HTTP status is already 200 because SSE headers precede generation. Signal the
                    # failure in-band and never manufacture a successful finish chunk.
                    event = {"error": {"message": "internal server error", "type": "server_error"}}
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                        self.wfile.write(b"data: [DONE]\n\n")
                    return
            if not streaming_free and not state["broken"]:
                # stream ended before a decision: unterminated CoT (thinking on) -> fallback; else flush buf
                if think == "enabled":
                    _emit(_THINK_FALLBACK)
                elif buf:
                    _emit(buf)
            if not state["broken"]:
                pt = ct = 0; fr = "stop"
                with contextlib.suppress(Exception):  # best-effort usage from the recorded result
                    out = j.result
                    pt = int(getattr(out, "prompt_tokens", 0) or 0)
                    ct = int(getattr(out, "generation_tokens", 0) or 0)
                    fr = getattr(out, "finish_reason", None) or "stop"
                fin = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": fr}],
                           usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(b"data: " + json.dumps(fin).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
        finally:
            for tp in tmp_paths:
                with contextlib.suppress(Exception): os.unlink(tp)

    # ---- legacy test format (nightly eval harness) ----
    def _legacy(self, req, prio=1):
        q = req.get("prompt", ""); mt = int(req.get("max_tokens", 256))
        temp = float(req.get("temp", 0.0)); think = req.get("thinking", "disabled")
        if isinstance(q, str) and q.lower().startswith("think:"):
            think = "enabled"; q = q[6:].lstrip()
        save_to = req.get("save_to"); tools = req.get("tools")
        try:
            # [auditfix] #1 (CRITICAL): was a direct _run() on this handler thread while _do_batched
            # ran generate_batch WITHOUT GEN_LOCK -> a POST / during a batched decode = two
            # simultaneous MLX generations on one Metal queue. Route through the worker queue like
            # _openai (the file's model; matches glm_serve_batched). Also lets legacy calls batch.
            out = submit_and_wait(q, mt, temp, think, tools, priority=prio)
        except Busy:
            self.send_response(503); self.send_header("Content-Type", "application/json")
            self.end_headers()
            with contextlib.suppress(Exception): self.wfile.write(b'{"error":"busy"}')
            return
        except Exception:
            # [auditfix] #5: there was NO generic except here — any error was a silent socket
            # drop (caller saw a connection reset, err.log stayed empty).
            _log_exc("_legacy")
            self.send_response(500); self.send_header("Content-Type", "application/json")
            self.end_headers()
            with contextlib.suppress(Exception):
                self.wfile.write(json.dumps({"error": "internal server error"}).encode())
            return
        text = getattr(out, "text", "") or ""
        text = strip_think(text, think)  # [auditfix] #15
        result = {"text": text, "tokens": getattr(out, "generation_tokens", None),
                  "prompt_tokens": getattr(out, "prompt_tokens", None),
                  "prompt_tps": getattr(out, "prompt_tps", None),
                  "tps": getattr(out, "generation_tps", None),
                  "finish": getattr(out, "finish_reason", None)}
        if save_to:
            ok, public_error = _save_text_create_only(save_to, text)
            if not ok:
                result["save_err"] = public_error
        with contextlib.suppress(Exception):
            body = json.dumps(result).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(body)

    def log_message(self, *a): pass


def _load():
    """Load the 256GB of weights. On ANY failure hard-exit non-zero so launchd KeepAlive relaunches
    (instead of an OOM-killed-mid-load process leaving the port bound but the model absent)."""
    global MODEL, PROC, CFG, READY
    print("loading M3 8-bit...", flush=True)
    try:
        MODEL, PROC = load(MD); CFG = load_config(MD)
    except BaseException as e:  # incl. MemoryError / interpreter-level failures
        print("FATAL load failed:", e, flush=True)
        _log_exc("_load")  # [auditfix] #5: full traceback to err.log before the hard exit
        os._exit(1)  # non-zero -> KeepAlive relaunch, not a silent wedge
    EOS_IDS.clear()
    _eos = getattr(CFG, "eos_token_id", None)
    if _eos is None and isinstance(CFG, dict): _eos = CFG.get("eos_token_id")
    if isinstance(_eos, int): EOS_IDS.add(_eos)
    elif isinstance(_eos, (list, tuple)): EOS_IDS.update(int(x) for x in _eos)
    if not EOS_IDS: EOS_IDS.add(200020)
    assert EOS_IDS, "EOS_IDS empty — batched path would never stop"  # [audit C2] belt-and-suspenders
    # warmup (increment-1 #5): compile the Metal decode kernels with one throwaway gen BEFORE we
    # flip READY, so the first real trade decision doesn't pay first-call kernel-compile latency.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _wp = apply_chat_template(PROC, CFG, [{"role": "user", "content": "ok"}], num_images=0, thinking_mode="disabled")
            generate(MODEL, PROC, _wp, max_tokens=2, temperature=0.0, verbose=True)
        print("warmup gen done", flush=True)
    except Exception as _we:
        print("warmup gen failed (non-fatal): %s" % _we, flush=True)
    threading.Thread(target=_batch_worker, daemon=True).start()
    READY = True   # flip READY only after warmup + worker up
    print("batching worker: max=%d window=%dms eos=%s" % (BATCH_MAX, BATCH_WINDOW_MS, sorted(EOS_IDS)), flush=True)
    print("LOADED — serving 127.0.0.1:%d" % PORT, flush=True)


def main():
    """Bind the localhost server and load the model. Imports/tests never execute this path."""
    if _RUNTIME_IMPORT_ERROR is not None or generate_batch is None:
        raise RuntimeError("M3 serving runtime dependencies are unavailable") from _RUNTIME_IMPORT_ERROR
    # Bind + start serving FIRST (port LISTENs in ms), THEN load the weights on the main thread.
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print("LISTENING 127.0.0.1:%d (loading weights, serving 503 until ready)" % PORT, flush=True)
    _load()
    # Block the main thread forever; the daemon serve thread handles requests.
    threading.Event().wait()


if __name__ == "__main__":
    main()
