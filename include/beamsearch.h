#ifndef BEAMSEARCH_H
#define BEAMSEARCH_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    beamsearch_status_ok = 0,
    beamsearch_status_invalid_argument,
    beamsearch_status_out_of_memory,
    beamsearch_status_output_too_small,
} beamsearch_status_t;

typedef struct {
    size_t beam_width;
    size_t token_cutoff;
    int32_t blank_id;
    /**
     * Number of worker threads used for beam expansion.
     * Use 0 or 1 for single-threaded decoding.
     */
    size_t thread_count;
} beamsearch_config_t;

typedef struct {
    size_t token_count;
    float log_score;
} beamsearch_result_t;

typedef struct beamsearch_decoder beamsearch_decoder_t;

/**
 * @brief Creates a reusable CTC beam-search decoder.
 * @return Owning pointer to a decoder or nullptr.
 */
[[nodiscard]]
beamsearch_decoder_t *beamsearch_decoder_create(
    const beamsearch_config_t *config);

void beamsearch_decoder_destroy(beamsearch_decoder_t *decoder);

/**
 * @brief Decodes frame-major CTC log probabilities.
 *
 * log_probs is read as log_probs[frame * vocab_size + token]. output_tokens is
 * caller-owned. Passing output_tokens as nullptr is valid only when
 * output_capacity is zero.
 */
[[nodiscard]]
beamsearch_status_t beamsearch_decoder_decode(
    beamsearch_decoder_t *decoder,
    const float *log_probs,
    size_t frame_count,
    size_t vocab_size,
    int32_t *output_tokens,
    size_t output_capacity,
    beamsearch_result_t *result);

#endif
