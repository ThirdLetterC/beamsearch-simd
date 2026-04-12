#ifndef BEAMSEARCH_CUDA_H
#define BEAMSEARCH_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct beamsearch_cuda_context beamsearch_cuda_context_t;

[[nodiscard]]
beamsearch_cuda_context_t *beamsearch_cuda_context_create(size_t max_frame_count,
                                                          size_t max_vocab_size,
                                                          size_t max_cutoff);

void beamsearch_cuda_context_destroy(beamsearch_cuda_context_t *context);

[[nodiscard]]
bool beamsearch_cuda_select_top_tokens(beamsearch_cuda_context_t *context,
                                       const float *frame,
                                       size_t vocab_size,
                                       size_t cutoff,
                                       int32_t blank_id,
                                       int32_t *tokens,
                                       float *scores,
                                       size_t *selected_count);

[[nodiscard]]
bool beamsearch_cuda_select_top_tokens_batch(beamsearch_cuda_context_t *context,
                                             const float *log_probs,
                                             size_t frame_count,
                                             size_t vocab_size,
                                             size_t cutoff,
                                             int32_t blank_id,
                                             int32_t *tokens,
                                             size_t *selected_counts);

#ifdef __cplusplus
}
#endif

#endif
