#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "beamsearch.h"
#ifdef BEAMSEARCH_ENABLE_CUDA
#include "beamsearch_cuda.h"
#endif
#include "psimd/psimd.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static constexpr int32_t BENCHMARK_BLANK_ID = 0;
static constexpr double NS_PER_SECOND = 1'000'000'000.0;
static constexpr uint64_t FNV_OFFSET = 14'695'981'039'346'656'037ULL;
static constexpr uint64_t FNV_PRIME = 1'099'511'628'211ULL;

typedef struct {
    const char *name;
    double audio_seconds;
    size_t sample_rate_hz;
    size_t frame_count;
    size_t vocab_size;
    size_t beam_width;
    size_t token_cutoff;
    size_t thread_count;
    size_t warmups;
    size_t repeats;
    uint32_t seed;
} benchmark_case_t;

typedef struct {
    double seconds;
    uint64_t checksum;
    beamsearch_result_t result;
} benchmark_measurement_t;

static void format_real_or_dash(char *buffer,
                                size_t capacity,
                                double value,
                                const char *format) {
    if (buffer == nullptr || capacity == 0) {
        return;
    }

    if (format == nullptr || !isfinite(value) || value <= 0.0) {
        auto written = snprintf(buffer, capacity, "-");
        if (written < 0 && capacity > 0) {
            buffer[0] = '\0';
        }
        return;
    }

    auto written = snprintf(buffer, capacity, format, value);
    if (written < 0) {
        buffer[0] = '\0';
        return;
    }
    if ((size_t)written >= capacity) {
        buffer[capacity - 1] = '\0';
    }
}

static void format_size_or_dash(char *buffer, size_t capacity, size_t value) {
    if (buffer == nullptr || capacity == 0) {
        return;
    }

    if (value == 0) {
        auto written = snprintf(buffer, capacity, "-");
        if (written < 0 && capacity > 0) {
            buffer[0] = '\0';
        }
        return;
    }

    auto written = snprintf(buffer, capacity, "%zu", value);
    if (written < 0) {
        buffer[0] = '\0';
        return;
    }
    if ((size_t)written >= capacity) {
        buffer[capacity - 1] = '\0';
    }
}

[[nodiscard]]
static const char *backend_name(psimd_backend backend) {
    switch (backend) {
    case psimd_backend_scalar:
        return "scalar";
    case psimd_backend_sse2:
        return "sse2";
    case psimd_backend_sse41:
        return "sse4.1";
    case psimd_backend_avx:
        return "avx";
    case psimd_backend_neon:
        return "neon";
    case psimd_backend_wasm:
        return "wasm";
    default:
        return "unknown";
    }
}

[[nodiscard]]
static uint64_t monotonic_ns() {
    struct timespec now = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0;
    }

    return (uint64_t)now.tv_sec * 1'000'000'000ULL + (uint64_t)now.tv_nsec;
}

[[nodiscard]]
static uint32_t rng_next(uint32_t *state) {
    auto value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

[[nodiscard]]
static float rng_unit(uint32_t *state) {
    return (float)(rng_next(state) & 0x00ffffffU) / 16'777'216.0f;
}

[[nodiscard]]
static int32_t nonblank_token_from_rank(size_t rank,
                                        int32_t blank_id,
                                        size_t vocab_size) {
    auto token = (int32_t)(rank % (vocab_size - 1));
    if (token >= blank_id) {
        token += 1;
    }
    return token;
}

[[nodiscard]]
static bool checked_mul_size(size_t lhs, size_t rhs, size_t *out) {
    if (out == nullptr) {
        return false;
    }
    if (lhs != 0 && rhs > SIZE_MAX / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

[[nodiscard]]
static bool generate_log_probs(float *log_probs,
                               size_t frame_count,
                               size_t vocab_size,
                               int32_t blank_id,
                               uint32_t seed) {
    if (log_probs == nullptr || vocab_size < 2 ||
        blank_id < 0 || (size_t)blank_id >= vocab_size) {
        return false;
    }

    auto rng = seed == 0 ? 1U : seed;
    for (size_t frame = 0; frame < frame_count; frame += 1) {
        auto row = log_probs + frame * vocab_size;
        auto emit_frame = (frame % 4 == 1) || (frame % 11 == 7);
        auto active = nonblank_token_from_rank(frame / 3 + seed, blank_id,
                                              vocab_size);
        auto confusable = nonblank_token_from_rank(frame / 3 + seed + 17,
                                                  blank_id, vocab_size);

        float max_logit = -INFINITY;
        for (size_t token = 0; token < vocab_size; token += 1) {
            float logit = -9.0f + 0.75f * rng_unit(&rng);
            if ((int32_t)token == blank_id) {
                logit = emit_frame ? 0.25f : 4.75f;
            } else if ((int32_t)token == active) {
                logit = emit_frame ? 4.5f : -1.25f;
            } else if ((int32_t)token == confusable) {
                logit = emit_frame ? 2.75f : -2.0f;
            }

            row[token] = logit;
            if (logit > max_logit) {
                max_logit = logit;
            }
        }

        auto exp_sum = 0.0f;
        for (size_t token = 0; token < vocab_size; token += 1) {
            exp_sum += expf(row[token] - max_logit);
        }

        auto log_sum = max_logit + logf(exp_sum);
        for (size_t token = 0; token < vocab_size; token += 1) {
            row[token] -= log_sum;
        }
    }

    return true;
}

[[nodiscard]]
static uint64_t hash_u64(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= FNV_PRIME;
    return hash;
}

[[nodiscard]]
static uint64_t hash_result(uint64_t hash,
                            const int32_t *tokens,
                            const beamsearch_result_t *result) {
    uint32_t score_bits = 0;
    memcpy(&score_bits, &result->log_score, sizeof(score_bits));

    hash = hash_u64(hash, result->token_count);
    hash = hash_u64(hash, score_bits);
    for (size_t i = 0; i < result->token_count; i += 1) {
        hash = hash_u64(hash, (uint32_t)tokens[i]);
    }
    return hash;
}

[[nodiscard]]
static bool run_repeated_decode(beamsearch_decoder_t *decoder,
                                const float *log_probs,
                                const benchmark_case_t *bench,
                                int32_t *output_tokens,
                                size_t repeats,
                                uint64_t *checksum,
                                beamsearch_result_t *last_result) {
    auto hash = FNV_OFFSET;
    for (size_t i = 0; i < repeats; i += 1) {
        beamsearch_result_t result = {0};
        auto status = beamsearch_decoder_decode(
            decoder, log_probs, bench->frame_count, bench->vocab_size,
            output_tokens, bench->frame_count, &result);
        if (status != beamsearch_status_ok) {
            fprintf(stderr, "decode failed for %s with status %d\n",
                    bench->name, (int)status);
            return false;
        }
        hash = hash_result(hash, output_tokens, &result);
        *last_result = result;
    }

    *checksum = hash;
    return true;
}

[[nodiscard]]
static bool measure_case(const benchmark_case_t *bench,
                         size_t repeat_scale,
                         benchmark_measurement_t *measurement) {
    if (bench == nullptr || measurement == nullptr || repeat_scale == 0) {
        return false;
    }

    size_t value_count = 0;
    if (!checked_mul_size(bench->frame_count, bench->vocab_size,
                          &value_count)) {
        return false;
    }

    float *log_probs = (float *)calloc(value_count, sizeof(float));
    int32_t *output_tokens = (int32_t *)calloc(bench->frame_count,
                                               sizeof(int32_t));
    if (log_probs == nullptr || output_tokens == nullptr) {
        free(log_probs);
        free(output_tokens);
        return false;
    }

    beamsearch_config_t config = {
        .beam_width = bench->beam_width,
        .token_cutoff = bench->token_cutoff,
        .thread_count = bench->thread_count,
        .blank_id = BENCHMARK_BLANK_ID,
    };
    auto decoder = beamsearch_decoder_create(&config);
    if (decoder == nullptr) {
        free(log_probs);
        free(output_tokens);
        return false;
    }

    if (!generate_log_probs(log_probs, bench->frame_count, bench->vocab_size,
                            BENCHMARK_BLANK_ID, bench->seed)) {
        beamsearch_decoder_destroy(decoder);
        free(log_probs);
        free(output_tokens);
        return false;
    }

    uint64_t ignored_checksum = 0;
    beamsearch_result_t ignored_result = {0};
    if (!run_repeated_decode(decoder, log_probs, bench, output_tokens,
                             bench->warmups, &ignored_checksum,
                             &ignored_result)) {
        beamsearch_decoder_destroy(decoder);
        free(log_probs);
        free(output_tokens);
        return false;
    }

    size_t repeats = 0;
    if (!checked_mul_size(bench->repeats, repeat_scale, &repeats) ||
        repeats == 0) {
        beamsearch_decoder_destroy(decoder);
        free(log_probs);
        free(output_tokens);
        return false;
    }

    auto start_ns = monotonic_ns();
    bool ok = run_repeated_decode(decoder, log_probs, bench, output_tokens,
                                  repeats, &measurement->checksum,
                                  &measurement->result);
    auto end_ns = monotonic_ns();
    if (!ok || end_ns <= start_ns) {
        beamsearch_decoder_destroy(decoder);
        free(log_probs);
        free(output_tokens);
        return false;
    }

    measurement->seconds = (double)(end_ns - start_ns) / NS_PER_SECOND;

    beamsearch_decoder_destroy(decoder);
    free(log_probs);
    free(output_tokens);
    return true;
}

static void print_header(size_t repeat_scale) {
    printf("ctc beamsearch benchmark\n");
    printf("backend: %s\n", backend_name(psimd_get_backend()));
    printf("repeat_scale: %zu\n\n", repeat_scale);
    printf("%-20s %7s %7s %7s %7s %7s %5s %6s %7s %8s %12s %9s %12s %12s "
           "%6s %11s %18s\n",
           "case", "audio_s", "sr_hz", "ctc_hz", "frames", "vocab", "beam",
           "cutoff", "threads", "repeats", "ms/decode", "rtf", "frames/s",
           "scores/s", "out", "score", "checksum");
}

static void print_case(const benchmark_case_t *bench,
                       size_t repeat_scale,
                       const benchmark_measurement_t *measurement) {
    auto repeats = bench->repeats * repeat_scale;
    auto seconds = measurement->seconds;
    auto ms_per_decode = seconds * 1'000.0 / (double)repeats;
    auto frames_per_second =
        ((double)bench->frame_count * (double)repeats) / seconds;
    auto scores_per_second =
        ((double)bench->frame_count * (double)bench->vocab_size *
         (double)repeats) /
        seconds;
    auto ctc_frames_per_audio_second =
        bench->audio_seconds > 0.0
            ? (double)bench->frame_count / bench->audio_seconds
            : 0.0;
    auto realtime_factor =
        bench->audio_seconds > 0.0
            ? (ms_per_decode / 1'000.0) / bench->audio_seconds
            : 0.0;

    char audio_seconds_text[16] = {0};
    char sample_rate_text[16] = {0};
    char ctc_rate_text[16] = {0};
    char realtime_factor_text[16] = {0};
    format_real_or_dash(audio_seconds_text, sizeof(audio_seconds_text),
                        bench->audio_seconds, "%.1f");
    format_size_or_dash(sample_rate_text, sizeof(sample_rate_text),
                        bench->sample_rate_hz);
    format_real_or_dash(ctc_rate_text, sizeof(ctc_rate_text),
                        ctc_frames_per_audio_second, "%.1f");
    format_real_or_dash(realtime_factor_text, sizeof(realtime_factor_text),
                        realtime_factor, "%.5f");

    printf("%-20s %7s %7s %7s %7zu %7zu %5zu %6zu %7zu %8zu %12.3f %9s "
           "%12.0f %12.0f %6zu %11.4f 0x%016" PRIx64 "\n",
           bench->name, audio_seconds_text, sample_rate_text, ctc_rate_text,
           bench->frame_count, bench->vocab_size, bench->beam_width,
           bench->token_cutoff, bench->thread_count, repeats, ms_per_decode,
           realtime_factor_text, frames_per_second, scores_per_second,
           measurement->result.token_count, measurement->result.log_score,
           measurement->checksum);
}

#ifdef BEAMSEARCH_ENABLE_CUDA
[[nodiscard]]
static uint64_t hash_cuda_selection(uint64_t hash,
                                    const int32_t *tokens,
                                    const size_t *selected_counts,
                                    size_t frame_count,
                                    size_t cutoff) {
    for (size_t frame = 0; frame < frame_count; frame += 1) {
        auto count = selected_counts[frame];
        hash = hash_u64(hash, count);
        for (size_t i = 0; i < count; i += 1) {
            hash = hash_u64(hash, (uint32_t)tokens[frame * cutoff + i]);
        }
    }
    return hash;
}

[[nodiscard]]
static bool measure_cuda_resident_select(const benchmark_case_t *bench,
                                         size_t repeat_scale,
                                         double *seconds_out,
                                         uint64_t *checksum_out) {
    if (bench == nullptr || seconds_out == nullptr ||
        checksum_out == nullptr || repeat_scale == 0 ||
        bench->token_cutoff == 0 || bench->token_cutoff > 128) {
        return false;
    }

    size_t value_count = 0;
    size_t selected_entry_count = 0;
    if (!checked_mul_size(bench->frame_count, bench->vocab_size,
                          &value_count) ||
        !checked_mul_size(bench->frame_count, bench->token_cutoff,
                          &selected_entry_count)) {
        return false;
    }

    auto log_probs = (float *)calloc(value_count, sizeof(float));
    auto tokens = (int32_t *)calloc(selected_entry_count, sizeof(int32_t));
    auto counts = (size_t *)calloc(bench->frame_count, sizeof(size_t));
    if (log_probs == nullptr || tokens == nullptr || counts == nullptr) {
        free(log_probs);
        free(tokens);
        free(counts);
        return false;
    }

    auto context = beamsearch_cuda_context_create(
        bench->frame_count, bench->vocab_size, bench->token_cutoff);
    if (context == nullptr ||
        !generate_log_probs(log_probs, bench->frame_count, bench->vocab_size,
                            BENCHMARK_BLANK_ID, bench->seed) ||
        !beamsearch_cuda_upload_log_probs(context, log_probs,
                                          bench->frame_count,
                                          bench->vocab_size)) {
        beamsearch_cuda_context_destroy(context);
        free(log_probs);
        free(tokens);
        free(counts);
        return false;
    }

    for (size_t i = 0; i < bench->warmups; i += 1) {
        if (!beamsearch_cuda_select_top_tokens_batch_device(
                context, bench->frame_count, bench->vocab_size,
                bench->token_cutoff, BENCHMARK_BLANK_ID, tokens, counts)) {
            beamsearch_cuda_context_destroy(context);
            free(log_probs);
            free(tokens);
            free(counts);
            return false;
        }
    }

    size_t repeats = 0;
    if (!checked_mul_size(bench->repeats, repeat_scale, &repeats) ||
        repeats == 0) {
        beamsearch_cuda_context_destroy(context);
        free(log_probs);
        free(tokens);
        free(counts);
        return false;
    }

    auto hash = FNV_OFFSET;
    auto start_ns = monotonic_ns();
    for (size_t i = 0; i < repeats; i += 1) {
        if (!beamsearch_cuda_select_top_tokens_batch_device(
                context, bench->frame_count, bench->vocab_size,
                bench->token_cutoff, BENCHMARK_BLANK_ID, tokens, counts)) {
            beamsearch_cuda_context_destroy(context);
            free(log_probs);
            free(tokens);
            free(counts);
            return false;
        }
        hash = hash_cuda_selection(hash, tokens, counts, bench->frame_count,
                                   bench->token_cutoff);
    }
    auto end_ns = monotonic_ns();
    if (end_ns <= start_ns) {
        beamsearch_cuda_context_destroy(context);
        free(log_probs);
        free(tokens);
        free(counts);
        return false;
    }

    *seconds_out = (double)(end_ns - start_ns) / NS_PER_SECOND;
    *checksum_out = hash;

    beamsearch_cuda_context_destroy(context);
    free(log_probs);
    free(tokens);
    free(counts);
    return true;
}

static void print_cuda_resident_select(const benchmark_case_t *cases,
                                       size_t case_count,
                                       size_t repeat_scale) {
    printf("\ncuda device-resident top-token selection\n");
    printf("%-20s %7s %7s %6s %8s %12s %12s %18s\n",
           "case", "frames", "vocab", "cutoff", "repeats", "ms/select",
           "scores/s", "checksum");

    for (size_t i = 0; i < case_count; i += 1) {
        if (cases[i].vocab_size < 65'536 || cases[i].token_cutoff > 128) {
            continue;
        }

        double seconds = 0.0;
        uint64_t checksum = 0;
        if (!measure_cuda_resident_select(&cases[i], repeat_scale, &seconds,
                                          &checksum)) {
            printf("%-20s %7zu %7zu %6zu %8s %12s %12s %18s\n",
                   cases[i].name, cases[i].frame_count, cases[i].vocab_size,
                   cases[i].token_cutoff, "-", "failed", "-", "-");
            continue;
        }

        auto repeats = cases[i].repeats * repeat_scale;
        auto ms_per_select = seconds * 1'000.0 / (double)repeats;
        auto scores_per_second =
            ((double)cases[i].frame_count * (double)cases[i].vocab_size *
             (double)repeats) /
            seconds;
        printf("%-20s %7zu %7zu %6zu %8zu %12.3f %12.0f 0x%016" PRIx64 "\n",
               cases[i].name, cases[i].frame_count, cases[i].vocab_size,
               cases[i].token_cutoff, repeats, ms_per_select,
               scores_per_second, checksum);
    }
}
#endif

[[nodiscard]]
static size_t parse_repeat_scale(int argc, char **argv) {
    if (argc < 2 || argv == nullptr || argv[1] == nullptr) {
        return 1;
    }

    char *end = nullptr;
    auto value = strtoull(argv[1], &end, 10);
    if (end == argv[1] || (end != nullptr && *end != '\0') || value == 0) {
        fprintf(stderr, "usage: %s [positive-repeat-scale]\n", argv[0]);
        return 0;
    }
    if (value > SIZE_MAX) {
        return 0;
    }
    return (size_t)value;
}

int main(int argc, char **argv) {
    const benchmark_case_t cases[] = {
        {
            .name = "asr-small-pruned",
            .frame_count = 128,
            .vocab_size = 128,
            .beam_width = 8,
            .token_cutoff = 16,
            .thread_count = 1,
            .warmups = 2,
            .repeats = 80,
            .seed = 11,
        },
        {
            .name = "asr-small-prun4",
            .frame_count = 128,
            .vocab_size = 128,
            .beam_width = 8,
            .token_cutoff = 16,
            .thread_count = 4,
            .warmups = 2,
            .repeats = 80,
            .seed = 11,
        },
        {
            .name = "asr-small-full1",
            .frame_count = 96,
            .vocab_size = 64,
            .beam_width = 8,
            .token_cutoff = 63,
            .thread_count = 1,
            .warmups = 2,
            .repeats = 30,
            .seed = 11,
        },
        {
            .name = "asr-small-full4",
            .frame_count = 96,
            .vocab_size = 64,
            .beam_width = 8,
            .token_cutoff = 63,
            .thread_count = 4,
            .warmups = 2,
            .repeats = 30,
            .seed = 11,
        },
        {
            .name = "asr-medium-1t",
            .frame_count = 300,
            .vocab_size = 1'024,
            .beam_width = 12,
            .token_cutoff = 32,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 18,
            .seed = 29,
        },
        {
            .name = "asr-medium-4t",
            .frame_count = 300,
            .vocab_size = 1'024,
            .beam_width = 12,
            .token_cutoff = 32,
            .thread_count = 4,
            .warmups = 1,
            .repeats = 18,
            .seed = 29,
        },
        {
            .name = "asr-wide-1t",
            .frame_count = 200,
            .vocab_size = 4'096,
            .beam_width = 12,
            .token_cutoff = 32,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 8,
            .seed = 47,
        },
        {
            .name = "asr-wide-4t",
            .frame_count = 200,
            .vocab_size = 4'096,
            .beam_width = 12,
            .token_cutoff = 32,
            .thread_count = 4,
            .warmups = 1,
            .repeats = 8,
            .seed = 47,
        },
        {
            .name = "30s-char-full",
            .audio_seconds = 30.0,
            .sample_rate_hz = 16'000,
            .frame_count = 1'500,
            .vocab_size = 64,
            .beam_width = 16,
            .token_cutoff = 63,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 2,
            .seed = 101,
        },
        {
            .name = "30s-wp1k-40ms",
            .audio_seconds = 30.0,
            .sample_rate_hz = 16'000,
            .frame_count = 750,
            .vocab_size = 1'024,
            .beam_width = 12,
            .token_cutoff = 32,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 3,
            .seed = 109,
        },
        {
            .name = "30s-wp1k-20ms",
            .audio_seconds = 30.0,
            .sample_rate_hz = 16'000,
            .frame_count = 1'500,
            .vocab_size = 1'024,
            .beam_width = 12,
            .token_cutoff = 32,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 2,
            .seed = 109,
        },
        {
            .name = "30s-bpe4k-20ms",
            .audio_seconds = 30.0,
            .sample_rate_hz = 16'000,
            .frame_count = 1'500,
            .vocab_size = 4'096,
            .beam_width = 12,
            .token_cutoff = 32,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 2,
            .seed = 127,
        },
        {
            .name = "30s-bpe8k-20ms",
            .audio_seconds = 30.0,
            .sample_rate_hz = 16'000,
            .frame_count = 1'500,
            .vocab_size = 8'192,
            .beam_width = 12,
            .token_cutoff = 32,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 1,
            .seed = 131,
        },
        {
            .name = "large-vocab-64k",
            .frame_count = 96,
            .vocab_size = 65'536,
            .beam_width = 12,
            .token_cutoff = 32,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 1,
            .seed = 151,
        },
        {
            .name = "xlarge-vocab-1m",
            .frame_count = 8,
            .vocab_size = 1'048'576,
            .beam_width = 4,
            .token_cutoff = 32,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 1,
            .seed = 157,
        },
        {
            .name = "30s-wp1k-b32-1",
            .audio_seconds = 30.0,
            .sample_rate_hz = 16'000,
            .frame_count = 1'500,
            .vocab_size = 1'024,
            .beam_width = 32,
            .token_cutoff = 64,
            .thread_count = 1,
            .warmups = 1,
            .repeats = 1,
            .seed = 137,
        },
        {
            .name = "30s-wp1k-b32-4",
            .audio_seconds = 30.0,
            .sample_rate_hz = 16'000,
            .frame_count = 1'500,
            .vocab_size = 1'024,
            .beam_width = 32,
            .token_cutoff = 64,
            .thread_count = 4,
            .warmups = 1,
            .repeats = 1,
            .seed = 137,
        },
    };

    auto repeat_scale = parse_repeat_scale(argc, argv);
    if (repeat_scale == 0) {
        return EXIT_FAILURE;
    }

    print_header(repeat_scale);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i += 1) {
        benchmark_measurement_t measurement = {0};
        if (!measure_case(&cases[i], repeat_scale, &measurement)) {
            fprintf(stderr, "benchmark case failed: %s\n", cases[i].name);
            return EXIT_FAILURE;
        }
        print_case(&cases[i], repeat_scale, &measurement);
    }

#ifdef BEAMSEARCH_ENABLE_CUDA
    print_cuda_resident_select(cases, sizeof(cases) / sizeof(cases[0]),
                               repeat_scale);
#endif

    printf("\nBuild once with default flags and once with -mavx2 -mfma to compare "
           "psimd backends.\n");
    printf("thread_count is a requested maximum; the decoder keeps small "
           "beam-expansion workloads on the single-thread path.\n");
    printf("30s rows assume 16 kHz PCM input feeding an acoustic model; this "
           "decoder consumes the post-model CTC log-prob frames shown in the "
           "table.\n");
    return EXIT_SUCCESS;
}
