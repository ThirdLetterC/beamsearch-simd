#include "beamsearch_cuda.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static constexpr size_t BEAMSEARCH_CUDA_MAX_CUTOFF = 128;
static constexpr size_t BEAMSEARCH_CUDA_MAX_BLOCKS = 1'024;
static constexpr size_t BEAMSEARCH_CUDA_MAX_GRID_FRAMES = 65'535;
static constexpr size_t BEAMSEARCH_CUDA_BATCH_THREADS = 64;
static constexpr size_t BEAMSEARCH_CUDA_TOKENS_PER_BLOCK = 256;
static constexpr float BEAMSEARCH_CUDA_LOG_ZERO = -INFINITY;

struct beamsearch_cuda_context {
    float *device_frame;
    int32_t *device_block_tokens;
    float *device_block_scores;
    size_t *device_block_counts;
    int32_t *device_tokens;
    float *device_scores;
    size_t *device_selected_count;
    int32_t *device_batch_tokens;
    size_t *device_batch_selected_counts;
    size_t max_frame_count;
    size_t max_vocab_size;
    size_t max_cutoff;
    size_t block_count;
};

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
static size_t cuda_block_count_for(size_t vocab_size) {
    auto block_count =
        (vocab_size + BEAMSEARCH_CUDA_TOKENS_PER_BLOCK - 1) /
        BEAMSEARCH_CUDA_TOKENS_PER_BLOCK;
    if (block_count == 0) {
        return 1;
    }
    if (block_count > BEAMSEARCH_CUDA_MAX_BLOCKS) {
        return BEAMSEARCH_CUDA_MAX_BLOCKS;
    }
    return block_count;
}

[[nodiscard]]
__device__ static bool cuda_token_score_better(float lhs_score,
                                               int32_t lhs_token,
                                               float rhs_score,
                                               int32_t rhs_token) {
    if (rhs_token < 0) {
        return true;
    }
    if (lhs_score > rhs_score) {
        return true;
    }
    if (lhs_score < rhs_score) {
        return false;
    }
    return lhs_token < rhs_token;
}

__device__ static void cuda_insert_top_token(int32_t *tokens,
                                             float *scores,
                                             size_t *selected_count,
                                             size_t cutoff,
                                             int32_t token,
                                             float score) {
    if (*selected_count == cutoff &&
        !cuda_token_score_better(score, token, scores[cutoff - 1],
                                 tokens[cutoff - 1])) {
        return;
    }

    auto insert_at = *selected_count;
    if (insert_at == cutoff) {
        insert_at -= 1;
    } else {
        *selected_count += 1;
    }

    while (insert_at > 0 &&
           cuda_token_score_better(score, token, scores[insert_at - 1],
                                   tokens[insert_at - 1])) {
        scores[insert_at] = scores[insert_at - 1];
        tokens[insert_at] = tokens[insert_at - 1];
        insert_at -= 1;
    }

    scores[insert_at] = score;
    tokens[insert_at] = token;
}

__global__ static void cuda_select_top_tokens_blocks(
    const float *log_probs,
    size_t frame_count,
    size_t vocab_size,
    size_t cutoff,
    int32_t blank_id,
    size_t block_count,
    int32_t *block_tokens,
    float *block_scores,
    size_t *block_counts) {
    if (threadIdx.x != 0) {
        return;
    }

    auto block_index = (size_t)blockIdx.x;
    auto frame_index = (size_t)blockIdx.y;
    if (frame_index >= frame_count) {
        return;
    }

    auto frame = log_probs + frame_index * vocab_size;
    auto begin = (block_index * vocab_size) / block_count;
    auto end = ((block_index + 1) * vocab_size) / block_count;
    auto token_base = (frame_index * block_count + block_index) * cutoff;

    for (size_t i = 0; i < cutoff; i += 1) {
        block_tokens[token_base + i] = -1;
        block_scores[token_base + i] = BEAMSEARCH_CUDA_LOG_ZERO;
    }

    size_t selected_count = 0;
    for (size_t token = begin; token < end; token += 1) {
        auto token_id = (int32_t)token;
        if (token_id != blank_id) {
            cuda_insert_top_token(block_tokens + token_base,
                                  block_scores + token_base,
                                  &selected_count, cutoff, token_id,
                                  frame[token]);
        }
    }

    block_counts[frame_index * block_count + block_index] = selected_count;
}

__global__ static void cuda_merge_top_tokens_blocks(
    const int32_t *block_tokens,
    const float *block_scores,
    const size_t *block_counts,
    size_t block_count,
    size_t cutoff,
    int32_t *tokens,
    float *scores,
    size_t *selected_count_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for (size_t i = 0; i < cutoff; i += 1) {
        tokens[i] = -1;
        scores[i] = BEAMSEARCH_CUDA_LOG_ZERO;
    }

    size_t selected_count = 0;
    for (size_t block_index = 0; block_index < block_count; block_index += 1) {
        auto block_base = block_index * cutoff;
        auto block_selected = block_counts[block_index];
        for (size_t i = 0; i < block_selected; i += 1) {
            cuda_insert_top_token(tokens, scores, &selected_count, cutoff,
                                  block_tokens[block_base + i],
                                  block_scores[block_base + i]);
        }
    }

    *selected_count_out = selected_count;
}

__global__ static void cuda_select_top_tokens_batch_kernel(
    const float *log_probs,
    size_t frame_count,
    size_t vocab_size,
    size_t cutoff,
    int32_t blank_id,
    int32_t *tokens,
    size_t *selected_counts) {
    extern __shared__ unsigned char shared_storage[];

    auto frame_index = (size_t)blockIdx.x;
    if (frame_index >= frame_count) {
        return;
    }

    auto thread_index = (size_t)threadIdx.x;
    auto thread_count = (size_t)blockDim.x;
    auto shared_tokens = (int32_t *)shared_storage;
    auto shared_scores = (float *)(shared_tokens + thread_count * cutoff);
    auto shared_counts = (size_t *)(shared_scores + thread_count * cutoff);
    auto local_tokens = shared_tokens + thread_index * cutoff;
    auto local_scores = shared_scores + thread_index * cutoff;

    for (size_t i = 0; i < cutoff; i += 1) {
        local_tokens[i] = -1;
        local_scores[i] = BEAMSEARCH_CUDA_LOG_ZERO;
    }
    shared_counts[thread_index] = 0;

    auto frame = log_probs + frame_index * vocab_size;
    for (size_t token = thread_index; token < vocab_size;
         token += thread_count) {
        auto token_id = (int32_t)token;
        if (token_id != blank_id) {
            cuda_insert_top_token(local_tokens, local_scores,
                                  &shared_counts[thread_index], cutoff,
                                  token_id, frame[token]);
        }
    }

    __syncthreads();

    if (thread_index != 0) {
        return;
    }

    auto output_tokens = tokens + frame_index * cutoff;
    float output_scores[BEAMSEARCH_CUDA_MAX_CUTOFF] = {0.0f};
    for (size_t i = 0; i < cutoff; i += 1) {
        output_tokens[i] = -1;
        output_scores[i] = BEAMSEARCH_CUDA_LOG_ZERO;
    }

    size_t selected_count = 0;
    for (size_t owner = 0; owner < thread_count; owner += 1) {
        auto owner_tokens = shared_tokens + owner * cutoff;
        auto owner_scores = shared_scores + owner * cutoff;
        auto owner_count = shared_counts[owner];
        for (size_t i = 0; i < owner_count; i += 1) {
            cuda_insert_top_token(output_tokens, output_scores,
                                  &selected_count, cutoff, owner_tokens[i],
                                  owner_scores[i]);
        }
    }

    selected_counts[frame_index] = selected_count;
}

beamsearch_cuda_context_t *beamsearch_cuda_context_create(size_t max_frame_count,
                                                          size_t max_vocab_size,
                                                          size_t max_cutoff) {
    if (max_frame_count == 0 ||
        max_frame_count > BEAMSEARCH_CUDA_MAX_GRID_FRAMES ||
        max_vocab_size == 0 || max_cutoff == 0 ||
        max_cutoff > BEAMSEARCH_CUDA_MAX_CUTOFF ||
        max_vocab_size > (size_t)INT32_MAX + 1) {
        return nullptr;
    }

    auto *context = (beamsearch_cuda_context_t *)calloc(
        1, sizeof(beamsearch_cuda_context_t));
    if (context == nullptr) {
        return nullptr;
    }

    context->max_frame_count = max_frame_count;
    context->max_vocab_size = max_vocab_size;
    context->max_cutoff = max_cutoff;
    context->block_count = cuda_block_count_for(max_vocab_size);

    size_t frame_bytes = 0;
    size_t block_result_count = 0;
    size_t top_entry_count = 0;
    size_t batch_entry_count = 0;
    size_t token_bytes = 0;
    size_t score_bytes = 0;
    size_t block_count_bytes = 0;
    size_t batch_token_bytes = 0;
    size_t batch_count_bytes = 0;
    if (!checked_mul_size(max_frame_count, max_vocab_size, &frame_bytes) ||
        !checked_mul_size(frame_bytes, sizeof(float), &frame_bytes) ||
        !checked_mul_size(max_frame_count, context->block_count,
                          &block_result_count) ||
        !checked_mul_size(block_result_count, max_cutoff,
                          &top_entry_count) ||
        !checked_mul_size(max_frame_count, max_cutoff, &batch_entry_count) ||
        !checked_mul_size(top_entry_count, sizeof(int32_t), &token_bytes) ||
        !checked_mul_size(top_entry_count, sizeof(float), &score_bytes) ||
        !checked_mul_size(block_result_count, sizeof(size_t),
                          &block_count_bytes) ||
        !checked_mul_size(batch_entry_count, sizeof(int32_t),
                          &batch_token_bytes) ||
        !checked_mul_size(max_frame_count, sizeof(size_t),
                          &batch_count_bytes)) {
        free(context);
        return nullptr;
    }

    if (cudaMalloc((void **)&context->device_frame, frame_bytes) !=
            cudaSuccess ||
        cudaMalloc((void **)&context->device_block_tokens, token_bytes) !=
            cudaSuccess ||
        cudaMalloc((void **)&context->device_block_scores, score_bytes) !=
            cudaSuccess ||
        cudaMalloc((void **)&context->device_block_counts, block_count_bytes) !=
            cudaSuccess ||
        cudaMalloc((void **)&context->device_tokens,
                   max_cutoff * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc((void **)&context->device_scores,
                   max_cutoff * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&context->device_selected_count,
                   sizeof(size_t)) != cudaSuccess ||
        cudaMalloc((void **)&context->device_batch_tokens,
                   batch_token_bytes) != cudaSuccess ||
        cudaMalloc((void **)&context->device_batch_selected_counts,
                   batch_count_bytes) != cudaSuccess) {
        beamsearch_cuda_context_destroy(context);
        return nullptr;
    }

    return context;
}

void beamsearch_cuda_context_destroy(beamsearch_cuda_context_t *context) {
    if (context == nullptr) {
        return;
    }

    (void)cudaFree(context->device_frame);
    (void)cudaFree(context->device_block_tokens);
    (void)cudaFree(context->device_block_scores);
    (void)cudaFree(context->device_block_counts);
    (void)cudaFree(context->device_tokens);
    (void)cudaFree(context->device_scores);
    (void)cudaFree(context->device_selected_count);
    (void)cudaFree(context->device_batch_tokens);
    (void)cudaFree(context->device_batch_selected_counts);
    free(context);
}

bool beamsearch_cuda_select_top_tokens(beamsearch_cuda_context_t *context,
                                       const float *frame,
                                       size_t vocab_size,
                                       size_t cutoff,
                                       int32_t blank_id,
                                       int32_t *tokens,
                                       float *scores,
                                       size_t *selected_count) {
    if (context == nullptr || frame == nullptr || tokens == nullptr ||
        scores == nullptr || selected_count == nullptr || vocab_size == 0 ||
        cutoff == 0 || vocab_size > context->max_vocab_size ||
        cutoff > context->max_cutoff || cutoff > BEAMSEARCH_CUDA_MAX_CUTOFF ||
        blank_id < 0 || (size_t)blank_id >= vocab_size ||
        vocab_size > (size_t)INT32_MAX + 1) {
        return false;
    }

    auto block_count = cuda_block_count_for(vocab_size);
    if (block_count > context->block_count) {
        return false;
    }

    auto frame_bytes = vocab_size * sizeof(float);
    auto result_token_bytes = cutoff * sizeof(int32_t);
    auto result_score_bytes = cutoff * sizeof(float);

    if (cudaMemcpy(context->device_frame, frame, frame_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    dim3 block_grid((unsigned int)block_count, 1U, 1U);
    cuda_select_top_tokens_blocks<<<block_grid, 1>>>(
        context->device_frame, 1, vocab_size, cutoff, blank_id, block_count,
        context->device_block_tokens, context->device_block_scores,
        context->device_block_counts);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    cuda_merge_top_tokens_blocks<<<1, 1>>>(
        context->device_block_tokens, context->device_block_scores,
        context->device_block_counts, block_count, cutoff,
        context->device_tokens, context->device_scores,
        context->device_selected_count);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    if (cudaMemcpy(selected_count, context->device_selected_count,
                   sizeof(size_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (*selected_count > cutoff) {
        return false;
    }
    if (*selected_count == 0) {
        return true;
    }

    if (cudaMemcpy(tokens, context->device_tokens, result_token_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(scores, context->device_scores, result_score_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool beamsearch_cuda_select_top_tokens_batch(beamsearch_cuda_context_t *context,
                                             const float *log_probs,
                                             size_t frame_count,
                                             size_t vocab_size,
                                             size_t cutoff,
                                             int32_t blank_id,
                                             int32_t *tokens,
                                             size_t *selected_counts) {
    if (!beamsearch_cuda_upload_log_probs(context, log_probs, frame_count,
                                          vocab_size)) {
        return false;
    }
    return beamsearch_cuda_select_top_tokens_batch_device(
        context, frame_count, vocab_size, cutoff, blank_id, tokens,
        selected_counts);
}

bool beamsearch_cuda_upload_log_probs(beamsearch_cuda_context_t *context,
                                      const float *log_probs,
                                      size_t frame_count,
                                      size_t vocab_size) {
    if (context == nullptr || log_probs == nullptr || frame_count == 0 ||
        vocab_size == 0 || frame_count > context->max_frame_count ||
        vocab_size > context->max_vocab_size) {
        return false;
    }

    size_t value_count = 0;
    size_t frame_bytes = 0;
    if (!checked_mul_size(frame_count, vocab_size, &value_count) ||
        !checked_mul_size(value_count, sizeof(float), &frame_bytes)) {
        return false;
    }

    if (cudaMemcpy(context->device_frame, log_probs, frame_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    return cudaDeviceSynchronize() == cudaSuccess;
}

bool beamsearch_cuda_select_top_tokens_batch_device(
    beamsearch_cuda_context_t *context,
    size_t frame_count,
    size_t vocab_size,
    size_t cutoff,
    int32_t blank_id,
    int32_t *tokens,
    size_t *selected_counts) {
    if (context == nullptr || tokens == nullptr || selected_counts == nullptr ||
        frame_count == 0 ||
        frame_count > BEAMSEARCH_CUDA_MAX_GRID_FRAMES || vocab_size == 0 ||
        cutoff == 0 || frame_count > context->max_frame_count ||
        vocab_size > context->max_vocab_size || cutoff > context->max_cutoff ||
        cutoff > BEAMSEARCH_CUDA_MAX_CUTOFF || blank_id < 0 ||
        (size_t)blank_id >= vocab_size ||
        vocab_size > (size_t)INT32_MAX + 1) {
        return false;
    }

    auto block_count = cuda_block_count_for(vocab_size);
    if (block_count > context->block_count) {
        return false;
    }

    size_t token_count = 0;
    size_t token_bytes = 0;
    size_t count_bytes = 0;
    if (!checked_mul_size(frame_count, cutoff, &token_count) ||
        !checked_mul_size(token_count, sizeof(int32_t), &token_bytes) ||
        !checked_mul_size(frame_count, sizeof(size_t), &count_bytes)) {
        return false;
    }

    auto shared_bytes =
        BEAMSEARCH_CUDA_BATCH_THREADS * cutoff *
            (sizeof(int32_t) + sizeof(float)) +
        BEAMSEARCH_CUDA_BATCH_THREADS * sizeof(size_t);
    cuda_select_top_tokens_batch_kernel<<<
        (unsigned int)frame_count, (unsigned int)BEAMSEARCH_CUDA_BATCH_THREADS,
        shared_bytes>>>(context->device_frame, frame_count, vocab_size, cutoff,
                        blank_id, context->device_batch_tokens,
                        context->device_batch_selected_counts);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    if (cudaMemcpy(tokens, context->device_batch_tokens, token_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(selected_counts, context->device_batch_selected_counts,
                   count_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    return cudaDeviceSynchronize() == cudaSuccess;
}
