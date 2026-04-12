#include "beamsearch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int g_failed = 0;

typedef struct {
    int32_t tokens[8];
    size_t length;
    float score;
} ref_hypothesis_t;

static void check(bool condition, const char *name) {
    if (!condition) {
        printf("FAIL %s\n", name);
        g_failed += 1;
        return;
    }
    printf("PASS %s\n", name);
}

static bool nearly_equal(float lhs, float rhs) {
    if (isinf(lhs) || isinf(rhs)) {
        return lhs == rhs;
    }
    return fabsf(lhs - rhs) < 1.0e-4f;
}

static float logaddexp_ref(float lhs, float rhs) {
    if (lhs == -INFINITY) {
        return rhs;
    }
    if (rhs == -INFINITY) {
        return lhs;
    }
    if (lhs < rhs) {
        auto tmp = lhs;
        lhs = rhs;
        rhs = tmp;
    }
    return lhs + log1pf(expf(rhs - lhs));
}

static bool same_tokens(const int32_t *lhs, size_t lhs_count,
                        const int32_t *rhs, size_t rhs_count) {
    if (lhs_count != rhs_count) {
        return false;
    }
    for (size_t i = 0; i < lhs_count; i += 1) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

static bool ref_better(const ref_hypothesis_t *lhs,
                       const ref_hypothesis_t *rhs) {
    if (lhs->score > rhs->score) {
        return true;
    }
    if (lhs->score < rhs->score) {
        return false;
    }

    auto count = lhs->length < rhs->length ? lhs->length : rhs->length;
    for (size_t i = 0; i < count; i += 1) {
        if (lhs->tokens[i] != rhs->tokens[i]) {
            return lhs->tokens[i] < rhs->tokens[i];
        }
    }
    return lhs->length < rhs->length;
}

static ref_hypothesis_t brute_force_ctc(const float *log_probs,
                                        size_t frame_count,
                                        size_t vocab_size,
                                        int32_t blank_id) {
    constexpr size_t max_hypotheses = 512;
    ref_hypothesis_t hypotheses[max_hypotheses] = {0};
    size_t hypothesis_count = 0;

    size_t path_count = 1;
    for (size_t i = 0; i < frame_count; i += 1) {
        path_count *= vocab_size;
    }

    for (size_t path_index = 0; path_index < path_count; path_index += 1) {
        int32_t collapsed[8] = {0};
        size_t collapsed_count = 0;
        auto score = 0.0f;
        auto encoded = path_index;
        int32_t previous = blank_id;

        for (size_t frame = 0; frame < frame_count; frame += 1) {
            auto token = (int32_t)(encoded % vocab_size);
            encoded /= vocab_size;
            score += log_probs[frame * vocab_size + (size_t)token];

            if (token == blank_id) {
                previous = blank_id;
                continue;
            }
            if (token != previous) {
                collapsed[collapsed_count] = token;
                collapsed_count += 1;
            }
            previous = token;
        }

        bool merged = false;
        for (size_t i = 0; i < hypothesis_count; i += 1) {
            if (same_tokens(hypotheses[i].tokens, hypotheses[i].length,
                            collapsed, collapsed_count)) {
                hypotheses[i].score = logaddexp_ref(hypotheses[i].score, score);
                merged = true;
                break;
            }
        }

        if (!merged && hypothesis_count < max_hypotheses) {
            for (size_t i = 0; i < collapsed_count; i += 1) {
                hypotheses[hypothesis_count].tokens[i] = collapsed[i];
            }
            hypotheses[hypothesis_count].length = collapsed_count;
            hypotheses[hypothesis_count].score = score;
            hypothesis_count += 1;
        }
    }

    ref_hypothesis_t best = hypotheses[0];
    for (size_t i = 1; i < hypothesis_count; i += 1) {
        if (ref_better(&hypotheses[i], &best)) {
            best = hypotheses[i];
        }
    }
    return best;
}

static beamsearch_decoder_t *new_decoder(size_t beam_width,
                                         size_t token_cutoff,
                                         int32_t blank_id) {
    beamsearch_config_t config = {
        .beam_width = beam_width,
        .token_cutoff = token_cutoff,
        .thread_count = 1,
        .blank_id = blank_id,
    };
    return beamsearch_decoder_create(&config);
}

static beamsearch_decoder_t *new_decoder_threads(size_t beam_width,
                                                 size_t token_cutoff,
                                                 int32_t blank_id,
                                                 size_t thread_count) {
    beamsearch_config_t config = {
        .beam_width = beam_width,
        .token_cutoff = token_cutoff,
        .thread_count = thread_count,
        .blank_id = blank_id,
    };
    return beamsearch_decoder_create(&config);
}

static void test_empty_input() {
    auto decoder = new_decoder(4, 2, 0);
    beamsearch_result_t result = {0};
    int32_t output[1] = {99};

    auto status = beamsearch_decoder_decode(decoder, nullptr, 0, 3, output, 1,
                                            &result);

    check(status == beamsearch_status_ok, "empty input status");
    check(result.token_count == 0 && result.log_score == 0.0f,
          "empty input result");
    beamsearch_decoder_destroy(decoder);
}

static void test_invalid_arguments() {
    beamsearch_config_t bad_config = {
        .beam_width = 0,
        .token_cutoff = 1,
        .blank_id = 0,
    };
    check(beamsearch_decoder_create(&bad_config) == nullptr,
          "invalid create rejects zero beam");

    auto decoder = new_decoder(2, 1, 0);
    beamsearch_result_t result = {0};
    float frame[2] = {logf(0.6f), logf(0.4f)};

    check(beamsearch_decoder_decode(nullptr, frame, 1, 2, nullptr, 0,
                                    &result) ==
              beamsearch_status_invalid_argument,
          "invalid null decoder");
    check(beamsearch_decoder_decode(decoder, frame, 1, 2, nullptr, 1,
                                    &result) ==
              beamsearch_status_invalid_argument,
          "invalid null output with capacity");
    check(beamsearch_decoder_decode(decoder, nullptr, 1, 2, nullptr, 0,
                                    &result) ==
              beamsearch_status_invalid_argument,
          "invalid null log probs");

    beamsearch_decoder_destroy(decoder);
}

static void test_repeated_token_collapse() {
    auto decoder = new_decoder(4, 1, 0);
    float log_probs[] = {
        logf(0.01f), logf(0.99f),
        logf(0.01f), logf(0.99f),
    };
    int32_t output[2] = {0};
    beamsearch_result_t result = {0};

    auto status = beamsearch_decoder_decode(decoder, log_probs, 2, 2, output, 2,
                                            &result);

    check(status == beamsearch_status_ok, "repeat collapse status");
    check(result.token_count == 1 && output[0] == 1,
          "repeat collapse output");
    beamsearch_decoder_destroy(decoder);
}

static void test_blank_separated_repeat() {
    auto decoder = new_decoder(6, 1, 0);
    float log_probs[] = {
        logf(0.01f), logf(0.99f),
        logf(0.99f), logf(0.01f),
        logf(0.01f), logf(0.99f),
    };
    int32_t output[3] = {0};
    beamsearch_result_t result = {0};

    auto status = beamsearch_decoder_decode(decoder, log_probs, 3, 2, output, 3,
                                            &result);

    check(status == beamsearch_status_ok, "blank repeat status");
    check(result.token_count == 2 && output[0] == 1 && output[1] == 1,
          "blank repeat output");
    beamsearch_decoder_destroy(decoder);
}

static void test_output_too_small() {
    auto decoder = new_decoder(6, 1, 0);
    float log_probs[] = {
        logf(0.01f), logf(0.99f),
        logf(0.99f), logf(0.01f),
        logf(0.01f), logf(0.99f),
    };
    int32_t output[1] = {77};
    beamsearch_result_t result = {0};

    auto status = beamsearch_decoder_decode(decoder, log_probs, 3, 2, output, 1,
                                            &result);

    check(status == beamsearch_status_output_too_small,
          "small output status");
    check(result.token_count == 2 && output[0] == 77,
          "small output required length");
    beamsearch_decoder_destroy(decoder);
}

static void test_reference_full_cutoff() {
    auto decoder = new_decoder(16, 2, 0);
    float log_probs[] = {
        logf(0.34f), logf(0.33f), logf(0.33f),
        logf(0.10f), logf(0.70f), logf(0.20f),
        logf(0.60f), logf(0.10f), logf(0.30f),
    };
    int32_t output[8] = {0};
    beamsearch_result_t result = {0};
    auto ref = brute_force_ctc(log_probs, 3, 3, 0);

    auto status = beamsearch_decoder_decode(decoder, log_probs, 3, 3, output, 8,
                                            &result);

    check(status == beamsearch_status_ok, "reference full status");
    check(same_tokens(output, result.token_count, ref.tokens, ref.length),
          "reference full tokens");
    check(nearly_equal(result.log_score, ref.score), "reference full score");
    beamsearch_decoder_destroy(decoder);
}

static void test_reference_pruned_cutoff() {
    auto decoder = new_decoder(16, 2, 0);
    float log_probs[] = {
        logf(0.10f), logf(0.50f), logf(0.40f), -INFINITY,
        logf(0.10f), logf(0.45f), logf(0.45f), -INFINITY,
        logf(0.50f), logf(0.25f), logf(0.25f), -INFINITY,
    };
    int32_t output[8] = {0};
    beamsearch_result_t result = {0};
    auto ref = brute_force_ctc(log_probs, 3, 4, 0);

    auto status = beamsearch_decoder_decode(decoder, log_probs, 3, 4, output, 8,
                                            &result);

    check(status == beamsearch_status_ok, "reference pruned status");
    check(same_tokens(output, result.token_count, ref.tokens, ref.length),
          "reference pruned tokens");
    check(nearly_equal(result.log_score, ref.score), "reference pruned score");
    beamsearch_decoder_destroy(decoder);
}

static void test_threaded_matches_single_thread() {
    auto single = new_decoder_threads(10, 4, 0, 1);
    auto threaded = new_decoder_threads(10, 4, 0, 4);
    float log_probs[] = {
        logf(0.10f), logf(0.50f), logf(0.20f), logf(0.10f), logf(0.10f),
        logf(0.10f), logf(0.10f), logf(0.45f), logf(0.25f), logf(0.10f),
        logf(0.65f), logf(0.10f), logf(0.10f), logf(0.10f), logf(0.05f),
        logf(0.10f), logf(0.50f), logf(0.10f), logf(0.25f), logf(0.05f),
        logf(0.20f), logf(0.10f), logf(0.10f), logf(0.50f), logf(0.10f),
    };
    int32_t single_output[8] = {0};
    int32_t threaded_output[8] = {0};
    beamsearch_result_t single_result = {0};
    beamsearch_result_t threaded_result = {0};

    auto single_status = beamsearch_decoder_decode(
        single, log_probs, 5, 5, single_output, 8, &single_result);
    auto threaded_status = beamsearch_decoder_decode(
        threaded, log_probs, 5, 5, threaded_output, 8, &threaded_result);

    check(single_status == beamsearch_status_ok &&
              threaded_status == beamsearch_status_ok,
          "threaded status");
    check(same_tokens(single_output, single_result.token_count,
                      threaded_output, threaded_result.token_count),
          "threaded tokens match");
    check(nearly_equal(single_result.log_score, threaded_result.log_score),
          "threaded score matches");

    beamsearch_decoder_destroy(single);
    beamsearch_decoder_destroy(threaded);
}

static void test_threaded_heavy_expansion_matches() {
    constexpr size_t frame_count = 4;
    constexpr size_t vocab_size = 65;
    auto single = new_decoder_threads(64, 64, 0, 1);
    auto threaded = new_decoder_threads(64, 64, 0, 4);
    float *log_probs = (float *)calloc(frame_count * vocab_size, sizeof(float));
    int32_t single_output[frame_count] = {0};
    int32_t threaded_output[frame_count] = {0};
    beamsearch_result_t single_result = {0};
    beamsearch_result_t threaded_result = {0};

    check(single != nullptr && threaded != nullptr && log_probs != nullptr,
          "threaded heavy setup");
    if (single == nullptr || threaded == nullptr || log_probs == nullptr) {
        beamsearch_decoder_destroy(single);
        beamsearch_decoder_destroy(threaded);
        free(log_probs);
        return;
    }

    for (size_t frame = 0; frame < frame_count; frame += 1) {
        for (size_t token = 0; token < vocab_size; token += 1) {
            log_probs[frame * vocab_size + token] = logf(0.001f);
        }
        log_probs[frame * vocab_size] = frame % 2 == 0 ? logf(0.40f)
                                                       : logf(0.05f);
        auto primary = (frame * 7 + 3) % (vocab_size - 1) + 1;
        auto secondary = (frame * 11 + 5) % (vocab_size - 1) + 1;
        log_probs[frame * vocab_size + primary] = logf(0.45f);
        log_probs[frame * vocab_size + secondary] = logf(0.20f);
    }

    auto single_status = beamsearch_decoder_decode(
        single, log_probs, frame_count, vocab_size, single_output, frame_count,
        &single_result);
    auto threaded_status = beamsearch_decoder_decode(
        threaded, log_probs, frame_count, vocab_size, threaded_output,
        frame_count, &threaded_result);

    check(single_status == beamsearch_status_ok &&
              threaded_status == beamsearch_status_ok,
          "threaded heavy status");
    check(same_tokens(single_output, single_result.token_count,
                      threaded_output, threaded_result.token_count),
          "threaded heavy tokens match");
    check(nearly_equal(single_result.log_score, threaded_result.log_score),
          "threaded heavy score matches");

    beamsearch_decoder_destroy(single);
    beamsearch_decoder_destroy(threaded);
    free(log_probs);
}

int main() {
    test_empty_input();
    test_invalid_arguments();
    test_repeated_token_collapse();
    test_blank_separated_repeat();
    test_output_too_small();
    test_reference_full_cutoff();
    test_reference_pruned_cutoff();
    test_threaded_matches_single_thread();
    test_threaded_heavy_expansion_matches();

    return g_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
