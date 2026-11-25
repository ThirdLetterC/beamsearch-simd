# beamsearch-simd

Strict C23 CTC prefix beam search decoder for ASR models.

The decoder consumes frame-major log probabilities:

```c
log_probs[frame * vocab_size + token]
```

It returns the single best collapsed token-id sequence and its log score. Version 1 has no language model, lexicon, tokenizer, string output, streaming state, or n-best output.

## Features

- Graves-style CTC prefix beam search in the log domain.
- psimd-accelerated per-frame token pruning.
- Optional CUDA per-frame token pruning for very large vocabularies.
- Parent-linked trie prefixes to avoid copying token arrays during search.
- Bounded trie: the child lookup table is rebuilt each frame from the live beams only, so it stays
  cache-resident (O(beam_width x token_cutoff)), and dead prefixes are compacted away once the
  trie grows past 4x its last live size (at least 65,536 nodes), keeping memory flat on long audio.
- Optional pthread worker path for larger beam expansion workloads.
- Caller-owned output buffer and reusable decoder scratch memory.

## Build And Test

This project uses strict warnings and C23/C2x. GCC 13 may require `-std=c2x`.

```bash
just check
```

Manual test build:

```bash
gcc -std=c2x -Wall -Wextra -Wpedantic -Werror -Iinclude -pthread \
    -fsanitize=address,undefined,leak \
    src/beamsearch.c tests/test_beamsearch.c -lm -o /tmp/beamsearch_test
/tmp/beamsearch_test
```

## Benchmark

Run the default and AVX2 benchmark builds:

```bash
just bench 1
just bench-avx2 1
```

Compare both backends:

```bash
just bench-compare 1
```

CUDA token-pruning builds require `nvcc` and define
`BEAMSEARCH_ENABLE_CUDA` for `src/beamsearch.c`:

```bash
just check-cuda
just bench-cuda 1
```

The CUDA path accelerates the top-token selection stage for large vocabularies
and falls back to the existing CPU selection path when CUDA is not compiled in,
when the vocabulary is below the CUDA threshold, or when a CUDA context cannot
be created.

`just bench-cuda` prints the normal end-to-end host API table and an additional
`cuda device-resident top-token selection` table. The second table uploads each
synthetic log-probability matrix once before timing and then repeatedly selects
top tokens from the resident device buffer, emulating an acoustic model that
already produced logits on the GPU.

The benchmark includes synthetic ASR shapes and 30-second, 16 kHz audio cases. The 30-second rows report `rtf`, where `0.01` means the decoder took about 1% of the audio duration.

## API

Public declarations are in `include/beamsearch.h`.

```c
beamsearch_config_t config = {
    .beam_width = 12,
    .token_cutoff = 32,
    .blank_id = 0,
    .thread_count = 1,
};

beamsearch_decoder_t *decoder = beamsearch_decoder_create(&config);
```

Destroy decoders with:

```c
beamsearch_decoder_destroy(decoder);
```
