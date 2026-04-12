# Security Model

This project is a C23 CTC prefix beam-search decoder. It consumes
caller-provided frame-major log probabilities and writes the single best
collapsed token-id sequence to a caller-owned output buffer.

## Trust Boundaries

- Public API inputs in `include/beamsearch.h` are untrusted until validated:
  `beamsearch_config_t`, `log_probs`, `frame_count`, `vocab_size`,
  `output_tokens`, `output_capacity`, and `beamsearch_result_t`.
- `log_probs` is treated as read-only model output. The library validates the
  pointer and shape arithmetic, but it does not authenticate, normalize, or
  range-check each floating-point value.
- `output_tokens` is caller-owned storage. The decoder writes to it only after
  it has computed the required token count and verified `output_capacity`.
- A `beamsearch_decoder_t` is an opaque, owning handle returned by
  `beamsearch_decoder_create` and released with `beamsearch_decoder_destroy`.
- `include/psimd/` is bundled third-party SIMD code and is part of the trusted
  computing base for vector loads, reductions, and backend selection.

## Attacker Model

An attacker may influence:

- Decoder configuration values such as `beam_width`, `token_cutoff`,
  `blank_id`, and `thread_count`.
- Decode dimensions such as `frame_count` and `vocab_size`.
- The contents of `log_probs`, including extreme finite values, infinities,
  NaNs, and highly pruned probability distributions.
- Allocation pressure through large beams, vocabularies, frame counts, and
  repeated decode calls.
- Threaded execution paths by selecting large enough workloads to trigger the
  pthread worker pool.

## Protected Assets

- Process memory safety while reading `log_probs`, maintaining decoder scratch
  buffers, expanding trie nodes, and writing `output_tokens`.
- Decoder state consistency across successful decodes, invalid arguments,
  allocation failures, and worker-pool startup failures.
- Service availability under malformed dimensions and memory pressure.
- Deterministic ownership of all memory allocated by `beamsearch_decoder_create`
  and grown during decode.

## Defensive Posture

- `beamsearch_decoder_create` fails closed for null configs, zero
  `beam_width`, zero `token_cutoff`, and negative `blank_id`.
- `beamsearch_decoder_decode` rejects null decoder/result pointers, zero
  vocabulary size, blank ids outside the active vocabulary, oversized
  vocabularies that cannot round-trip through `int32_t` token ids, overflowing
  `frame_count * vocab_size` shape arithmetic, null `log_probs` for non-empty
  input, and null output buffers with non-zero capacity.
- Empty input is accepted explicitly: `frame_count == 0` returns success with
  `token_count == 0` and `log_score == 0.0f`.
- Internal allocation size calculations use checked `size_t` multiplication and
  addition before allocating or indexing derived buffers.
- Internal dynamic arrays grow by checked doubling and fail closed if capacity
  growth would overflow.
- Allocations use zero-initializing `calloc` through `calloc_array`; allocation
  failure returns `nullptr` or `beamsearch_status_out_of_memory`.
- Decode scratch buffers are owned by the decoder and reused between calls.
  Failed allocations do not expose partially written output as a successful
  decode.
- Trie child and accumulator hash tables maintain explicit occupancy/epoch
  metadata and are rebuilt through fresh zeroed storage when capacity grows.
- Top-token selection excludes the configured blank token and clamps
  `token_cutoff` to the number of nonblank tokens available in the vocabulary.
- Output reconstruction is delayed until after the best prefix length is known.
  If `output_capacity` is too small, the function returns
  `beamsearch_status_output_too_small` and reports the required length in
  `result->token_count`.
- The optional pthread worker path precomputes shared trie children before
  worker expansion. Workers write to per-worker transition/select buffers, and
  the owning decoder thread merges results after synchronization.
- `thread_count` is normalized at decoder creation: 0 and 1 select the
  single-thread path, and larger values are capped to a fixed internal worker
  limit before any worker storage is allocated or pthreads are created.
- `beamsearch_decoder_destroy` stops and joins worker threads before freeing
  decoder-owned buffers.
- Public APIs return explicit status values rather than partial-success
  semantics.

## Limitations And Caller Responsibilities

- A single decoder handle is not safe for concurrent API calls. Callers must
  serialize access to each `beamsearch_decoder_t` or create one decoder per
  thread.
- Caller pointers must be valid for their declared capacities. Passing stale,
  dangling, undersized, or otherwise invalid pointers is out of scope.
- `log_probs` must contain at least `frame_count * vocab_size` readable `float`
  values laid out as `log_probs[frame * vocab_size + token]`.
- Scores are assumed to be log probabilities or compatible log-domain scores.
  NaNs and unusual infinities may affect ranking semantics; callers should
  sanitize model output when those values are security-relevant.
- The decoder returns token ids only. It does not validate tokenizer, lexicon,
  Unicode, or application-level text safety because version 1 has no tokenizer,
  language model, string output, streaming state, or n-best output.
- This library does not zeroize decoder scratch memory on destroy. Do not store
  secrets in token or probability buffers unless the embedding application owns
  the required wiping policy.
- `thread_count` controls internal worker parallelism only. It is a requested
  maximum, not a guarantee that the decoder will create that many workers. The
  public decoder object remains mutable shared state and requires external
  synchronization.
- `include/psimd/` is vendored code. Keep local changes isolated and review
  backend-specific intrinsic changes separately.

## Build Hardening

- The checked build path uses C23/C2x mode with strict warnings:
  `-std=c2x -Wall -Wextra -Wpedantic -Werror`.
- `just check` compiles the decoder, runs the sanitizer-backed unit tests, runs
  an AVX2 test build, and compiles the bundled psimd test.
- `just test` builds the unit tests with
  `-fsanitize=address,undefined,leak`. Downstream debug builds should add
  `-fno-omit-frame-pointer` when stack traces are required.
- `just test-avx2` exercises the AVX2/FMA code generation path with
  `-mavx2 -mfma`.
- Production embedders should add their platform hardening flags around the
  library build, such as `-fstack-protector-strong`, `_FORTIFY_SOURCE` where
  supported by the optimization level and libc, PIE, and RELRO/now linking.

## Reporting

Report security issues with:

- A minimal reproducer or input dimensions.
- Compiler, target architecture, selected psimd backend, and build flags.
- Whether the issue reproduces under `just test`, `just test-avx2`, or a custom
  sanitizer build.
