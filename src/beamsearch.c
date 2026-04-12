#include "beamsearch.h"

#ifdef BEAMSEARCH_ENABLE_CUDA
#include "beamsearch_cuda.h"
#endif

#include "psimd/psimd.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static constexpr float BEAMSEARCH_LOG_ZERO = -INFINITY;
static constexpr size_t BEAMSEARCH_INVALID_NODE = SIZE_MAX;
static constexpr size_t BEAMSEARCH_MIN_HASH_CAPACITY = 16;
static constexpr size_t BEAMSEARCH_MIN_THREADED_SELECT_TOKENS = 16'384;
static constexpr size_t BEAMSEARCH_MIN_THREADED_EXPANSIONS = 2'048;
static constexpr size_t BEAMSEARCH_MAX_WORKER_THREADS = 64;
#ifdef BEAMSEARCH_ENABLE_CUDA
static constexpr size_t BEAMSEARCH_MIN_CUDA_SELECT_TOKENS = 65'536;
static constexpr size_t BEAMSEARCH_MAX_CUDA_SELECT_CUTOFF = 128;
#endif
static constexpr uint32_t BEAMSEARCH_INITIAL_EPOCH = 1;

typedef struct {
    size_t parent;
    int32_t token;
    size_t length;
} beamsearch_trie_node_t;

typedef struct {
    size_t node;
    float p_blank;
    float p_nonblank;
    float score;
} beamsearch_beam_t;

typedef struct {
    size_t parent;
    int32_t token;
    size_t child;
    bool occupied;
} beamsearch_child_slot_t;

typedef struct {
    size_t node;
    float p_blank;
    float p_nonblank;
    uint32_t epoch;
} beamsearch_accum_slot_t;

typedef struct {
    size_t node;
    float value;
    bool nonblank;
} beamsearch_transition_t;

typedef enum {
    beamsearch_worker_task_select = 0,
    beamsearch_worker_task_expand = 1,
} beamsearch_worker_task_t;

typedef struct {
    pthread_t thread;
    beamsearch_decoder_t *decoder;
    size_t worker_index;
    uint64_t seen_generation;
    const float *frame;
    size_t vocab_begin;
    size_t vocab_end;
    size_t beam_begin;
    size_t beam_end;
    size_t selected_count;
    int32_t *local_tokens;
    float *local_scores;
    size_t local_count;
    size_t local_capacity;
    beamsearch_transition_t *transitions;
    size_t transition_count;
    size_t transition_capacity;
    bool ok;
} beamsearch_worker_t;

struct beamsearch_decoder {
    beamsearch_config_t config;
    size_t thread_count;

    beamsearch_trie_node_t *nodes;
    size_t node_count;
    size_t node_capacity;

    beamsearch_child_slot_t *child_slots;
    size_t child_count;
    size_t child_capacity;

    beamsearch_accum_slot_t *accum_slots;
    size_t accum_count;
    size_t accum_capacity;
    uint32_t accum_epoch;

    beamsearch_beam_t *current_beams;
    size_t current_count;
    size_t current_capacity;

    beamsearch_beam_t *next_beams;
    size_t next_count;
    size_t next_capacity;

    int32_t *selected_tokens;
    size_t selected_token_capacity;

    float *selected_scores;
    size_t selected_score_capacity;

    size_t *expanded_children;
    size_t expanded_child_capacity;

    beamsearch_worker_t *workers;
    size_t worker_capacity;
    size_t worker_count;
    size_t active_worker_count;
    size_t workers_done;
    uint64_t worker_generation;
    beamsearch_worker_task_t worker_task;
    pthread_mutex_t worker_mutex;
    pthread_cond_t worker_start;
    pthread_cond_t worker_done;
    bool worker_sync_ready;
    bool workers_started;
    bool worker_stop;

#ifdef BEAMSEARCH_ENABLE_CUDA
    beamsearch_cuda_context_t *cuda_context;
    int32_t *cuda_selected_tokens;
    size_t *cuda_selected_counts;
    size_t cuda_selected_frame_capacity;
    size_t cuda_selected_cutoff_capacity;
    size_t cuda_frame_capacity;
    size_t cuda_vocab_capacity;
    size_t cuda_cutoff_capacity;
#endif
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
static bool checked_add_size(size_t lhs, size_t rhs, size_t *out) {
    if (out == nullptr) {
        return false;
    }
    if (rhs > SIZE_MAX - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

[[nodiscard]]
static void *calloc_array(size_t count, size_t element_size) {
    size_t byte_count = 0;
    if (!checked_mul_size(count, element_size, &byte_count)) {
        return nullptr;
    }
    if (byte_count == 0) {
        return nullptr;
    }
    return calloc(count, element_size);
}

[[nodiscard]]
static size_t grow_capacity(size_t current, size_t required) {
    auto capacity = current;
    if (capacity == 0) {
        capacity = BEAMSEARCH_MIN_HASH_CAPACITY;
    }
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2) {
            return 0;
        }
        capacity *= 2;
    }
    return capacity;
}

[[nodiscard]]
static size_t sanitize_thread_count(size_t requested) {
    if (requested < 2) {
        return 1;
    }
    if (requested > BEAMSEARCH_MAX_WORKER_THREADS) {
        return BEAMSEARCH_MAX_WORKER_THREADS;
    }
    return requested;
}

static void partition_range(size_t count,
                            size_t parts,
                            size_t index,
                            size_t *begin_out,
                            size_t *end_out) {
    if (parts == 0 || index >= parts) {
        *begin_out = 0;
        *end_out = 0;
        return;
    }

    auto base = count / parts;
    auto remainder = count % parts;
    auto extra_before = index < remainder ? index : remainder;
    auto begin = index * base + extra_before;

    *begin_out = begin;
    *end_out = begin + base + (index < remainder ? 1 : 0);
}

#ifdef BEAMSEARCH_ENABLE_CUDA
[[nodiscard]]
static bool ensure_cuda_context(beamsearch_decoder_t *decoder,
                                size_t frame_count,
                                size_t vocab_size,
                                size_t cutoff) {
    if (cutoff > BEAMSEARCH_MAX_CUDA_SELECT_CUTOFF) {
        return false;
    }
    if (decoder->cuda_context != nullptr &&
        frame_count <= decoder->cuda_frame_capacity &&
        vocab_size <= decoder->cuda_vocab_capacity &&
        cutoff <= decoder->cuda_cutoff_capacity) {
        return true;
    }

    beamsearch_cuda_context_destroy(decoder->cuda_context);
    decoder->cuda_context =
        beamsearch_cuda_context_create(frame_count, vocab_size, cutoff);
    if (decoder->cuda_context == nullptr) {
        decoder->cuda_frame_capacity = 0;
        decoder->cuda_vocab_capacity = 0;
        decoder->cuda_cutoff_capacity = 0;
        return false;
    }

    decoder->cuda_frame_capacity = frame_count;
    decoder->cuda_vocab_capacity = vocab_size;
    decoder->cuda_cutoff_capacity = cutoff;
    return true;
}

[[nodiscard]]
static bool ensure_cuda_selected_capacity(beamsearch_decoder_t *decoder,
                                          size_t frame_count,
                                          size_t cutoff) {
    if (frame_count <= decoder->cuda_selected_frame_capacity &&
        cutoff <= decoder->cuda_selected_cutoff_capacity) {
        return true;
    }

    size_t entry_count = 0;
    if (!checked_mul_size(frame_count, cutoff, &entry_count)) {
        return false;
    }

    auto tokens = (int32_t *)calloc_array(entry_count, sizeof(int32_t));
    if (tokens == nullptr) {
        return false;
    }
    auto counts = (size_t *)calloc_array(frame_count, sizeof(size_t));
    if (counts == nullptr) {
        free(tokens);
        return false;
    }

    free(decoder->cuda_selected_tokens);
    free(decoder->cuda_selected_counts);
    decoder->cuda_selected_tokens = tokens;
    decoder->cuda_selected_counts = counts;
    decoder->cuda_selected_frame_capacity = frame_count;
    decoder->cuda_selected_cutoff_capacity = cutoff;
    return true;
}
#endif

[[nodiscard]]
static bool hash_capacity_for(size_t required_entries, size_t *capacity_out) {
    size_t required_slots = 0;
    if (!checked_mul_size(required_entries, 2, &required_slots)) {
        return false;
    }
    if (required_slots < BEAMSEARCH_MIN_HASH_CAPACITY) {
        required_slots = BEAMSEARCH_MIN_HASH_CAPACITY;
    }

    size_t capacity = BEAMSEARCH_MIN_HASH_CAPACITY;
    while (capacity < required_slots) {
        if (capacity > SIZE_MAX / 2) {
            return false;
        }
        capacity *= 2;
    }

    *capacity_out = capacity;
    return true;
}

[[nodiscard]]
static uint64_t mix_u64(uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

[[nodiscard]]
static size_t hash_node(size_t node) {
    return (size_t)mix_u64((uint64_t)node);
}

[[nodiscard]]
static size_t hash_child(size_t parent, int32_t token) {
    uint64_t value = (uint64_t)parent;
    value ^= ((uint64_t)(uint32_t)token + 0x9e3779b97f4a7c15ULL +
              (value << 6) + (value >> 2));
    return (size_t)mix_u64(value);
}

[[nodiscard]]
static float logaddexp_pair(float lhs, float rhs) {
    if (lhs == BEAMSEARCH_LOG_ZERO) {
        return rhs;
    }
    if (rhs == BEAMSEARCH_LOG_ZERO) {
        return lhs;
    }
    if (lhs < rhs) {
        auto tmp = lhs;
        lhs = rhs;
        rhs = tmp;
    }
    return lhs + log1pf(expf(rhs - lhs));
}

[[nodiscard]]
static float beam_total_score(const beamsearch_beam_t *beam) {
    return logaddexp_pair(beam->p_blank, beam->p_nonblank);
}

[[nodiscard]]
static bool beam_better(const beamsearch_beam_t *lhs,
                        const beamsearch_beam_t *rhs) {
    if (lhs->score > rhs->score) {
        return true;
    }
    if (lhs->score < rhs->score) {
        return false;
    }
    return lhs->node < rhs->node;
}

[[nodiscard]]
static bool beam_worse(const beamsearch_beam_t *lhs,
                       const beamsearch_beam_t *rhs) {
    return beam_better(rhs, lhs);
}

static void swap_beams(beamsearch_beam_t *lhs, beamsearch_beam_t *rhs) {
    auto tmp = *lhs;
    *lhs = *rhs;
    *rhs = tmp;
}

static void heap_sift_up(beamsearch_beam_t *heap, size_t index) {
    while (index > 0) {
        auto parent = (index - 1) / 2;
        if (!beam_worse(&heap[index], &heap[parent])) {
            break;
        }
        swap_beams(&heap[index], &heap[parent]);
        index = parent;
    }
}

static void heap_sift_down(beamsearch_beam_t *heap, size_t count,
                           size_t index) {
    while (true) {
        auto left = index * 2 + 1;
        auto right = left + 1;
        auto worst = index;

        if (left < count && beam_worse(&heap[left], &heap[worst])) {
            worst = left;
        }
        if (right < count && beam_worse(&heap[right], &heap[worst])) {
            worst = right;
        }
        if (worst == index) {
            break;
        }
        swap_beams(&heap[index], &heap[worst]);
        index = worst;
    }
}

static void heap_consider(beamsearch_beam_t *heap, size_t *count,
                          size_t capacity, beamsearch_beam_t candidate) {
    if (*count < capacity) {
        heap[*count] = candidate;
        heap_sift_up(heap, *count);
        *count += 1;
        return;
    }
    if (capacity != 0 && beam_better(&candidate, &heap[0])) {
        heap[0] = candidate;
        heap_sift_down(heap, *count, 0);
    }
}

static void sort_beams_desc(beamsearch_beam_t *beams, size_t count) {
    for (size_t i = 1; i < count; i += 1) {
        auto value = beams[i];
        auto j = i;
        while (j > 0 && beam_better(&value, &beams[j - 1])) {
            beams[j] = beams[j - 1];
            j -= 1;
        }
        beams[j] = value;
    }
}

[[nodiscard]]
static bool ensure_node_capacity(beamsearch_decoder_t *decoder,
                                 size_t required) {
    if (required <= decoder->node_capacity) {
        return true;
    }

    auto capacity = grow_capacity(decoder->node_capacity, required);
    if (capacity == 0) {
        return false;
    }

    auto grown = (beamsearch_trie_node_t *)calloc_array(
        capacity, sizeof(decoder->nodes[0]));
    if (grown == nullptr) {
        return false;
    }
    if (decoder->node_count != 0) {
        memcpy(grown, decoder->nodes,
               decoder->node_count * sizeof(decoder->nodes[0]));
    }
    free(decoder->nodes);
    decoder->nodes = grown;
    decoder->node_capacity = capacity;
    return true;
}

[[nodiscard]]
static bool ensure_beam_capacity(beamsearch_beam_t **beams,
                                 size_t *capacity_in_out,
                                 size_t current_count,
                                 size_t required) {
    if (required <= *capacity_in_out) {
        return true;
    }

    auto capacity = grow_capacity(*capacity_in_out, required);
    if (capacity == 0) {
        return false;
    }

    auto grown = (beamsearch_beam_t *)calloc_array(capacity, sizeof((*beams)[0]));
    if (grown == nullptr) {
        return false;
    }
    if (current_count != 0) {
        memcpy(grown, *beams, current_count * sizeof((*beams)[0]));
    }
    free(*beams);
    *beams = grown;
    *capacity_in_out = capacity;
    return true;
}

[[nodiscard]]
static bool ensure_selected_token_capacity(beamsearch_decoder_t *decoder,
                                           size_t required) {
    if (required <= decoder->selected_token_capacity) {
        return true;
    }

    auto capacity = grow_capacity(decoder->selected_token_capacity, required);
    if (capacity == 0) {
        return false;
    }

    auto grown = (int32_t *)calloc_array(capacity,
                                        sizeof(decoder->selected_tokens[0]));
    if (grown == nullptr) {
        return false;
    }
    free(decoder->selected_tokens);
    decoder->selected_tokens = grown;
    decoder->selected_token_capacity = capacity;
    return true;
}

[[nodiscard]]
static bool ensure_selected_score_capacity(beamsearch_decoder_t *decoder,
                                           size_t required) {
    if (required <= decoder->selected_score_capacity) {
        return true;
    }

    auto capacity = grow_capacity(decoder->selected_score_capacity, required);
    if (capacity == 0) {
        return false;
    }

    auto grown = (float *)calloc_array(capacity,
                                      sizeof(decoder->selected_scores[0]));
    if (grown == nullptr) {
        return false;
    }
    free(decoder->selected_scores);
    decoder->selected_scores = grown;
    decoder->selected_score_capacity = capacity;
    return true;
}

[[nodiscard]]
static bool ensure_expanded_child_capacity(beamsearch_decoder_t *decoder,
                                           size_t required) {
    if (required <= decoder->expanded_child_capacity) {
        return true;
    }

    auto capacity = grow_capacity(decoder->expanded_child_capacity, required);
    if (capacity == 0) {
        return false;
    }

    auto grown = (size_t *)calloc_array(capacity,
                                       sizeof(decoder->expanded_children[0]));
    if (grown == nullptr) {
        return false;
    }
    free(decoder->expanded_children);
    decoder->expanded_children = grown;
    decoder->expanded_child_capacity = capacity;
    return true;
}

[[nodiscard]]
static bool ensure_worker_capacity(beamsearch_decoder_t *decoder,
                                   size_t required) {
    if (required <= decoder->worker_capacity) {
        return true;
    }

    auto capacity = grow_capacity(decoder->worker_capacity, required);
    if (capacity == 0) {
        return false;
    }

    auto grown = (beamsearch_worker_t *)calloc_array(capacity,
                                                    sizeof(beamsearch_worker_t));
    if (grown == nullptr) {
        return false;
    }
    if (decoder->worker_capacity != 0) {
        memcpy(grown, decoder->workers,
               decoder->worker_capacity * sizeof(decoder->workers[0]));
    }
    free(decoder->workers);
    decoder->workers = grown;
    decoder->worker_capacity = capacity;
    return true;
}

[[nodiscard]]
static bool ensure_worker_transition_capacity(beamsearch_worker_t *worker,
                                              size_t required) {
    if (required <= worker->transition_capacity) {
        return true;
    }

    auto capacity = grow_capacity(worker->transition_capacity, required);
    if (capacity == 0) {
        return false;
    }

    auto grown = (beamsearch_transition_t *)calloc_array(capacity,
                                                        sizeof(beamsearch_transition_t));
    if (grown == nullptr) {
        return false;
    }
    free(worker->transitions);
    worker->transitions = grown;
    worker->transition_capacity = capacity;
    return true;
}

[[nodiscard]]
static bool ensure_worker_select_capacity(beamsearch_worker_t *worker,
                                          size_t required) {
    if (required <= worker->local_capacity) {
        return true;
    }

    auto capacity = grow_capacity(worker->local_capacity, required);
    if (capacity == 0) {
        return false;
    }

    int32_t *tokens = (int32_t *)calloc_array(capacity, sizeof(int32_t));
    if (tokens == nullptr) {
        return false;
    }
    float *scores = (float *)calloc_array(capacity, sizeof(float));
    if (scores == nullptr) {
        free(tokens);
        return false;
    }

    free(worker->local_tokens);
    free(worker->local_scores);
    worker->local_tokens = tokens;
    worker->local_scores = scores;
    worker->local_capacity = capacity;
    return true;
}

static void child_insert_existing(beamsearch_child_slot_t *slots,
                                  size_t capacity,
                                  beamsearch_child_slot_t slot) {
    auto index = hash_child(slot.parent, slot.token) & (capacity - 1);
    while (slots[index].occupied) {
        index = (index + 1) & (capacity - 1);
    }
    slots[index] = slot;
}

[[nodiscard]]
static bool ensure_child_capacity(beamsearch_decoder_t *decoder,
                                  size_t required_entries) {
    size_t capacity = 0;
    if (!hash_capacity_for(required_entries, &capacity)) {
        return false;
    }
    if (capacity <= decoder->child_capacity) {
        return true;
    }

    beamsearch_child_slot_t *slots = (beamsearch_child_slot_t *)calloc_array(
        capacity, sizeof(beamsearch_child_slot_t));
    if (slots == nullptr) {
        return false;
    }

    for (size_t i = 0; i < decoder->child_capacity; i += 1) {
        if (decoder->child_slots[i].occupied) {
            child_insert_existing(slots, capacity, decoder->child_slots[i]);
        }
    }

    free(decoder->child_slots);
    decoder->child_slots = slots;
    decoder->child_capacity = capacity;
    return true;
}

static void accum_insert_existing(beamsearch_accum_slot_t *slots,
                                  size_t capacity,
                                  beamsearch_accum_slot_t slot) {
    auto index = hash_node(slot.node) & (capacity - 1);
    while (slots[index].epoch == slot.epoch) {
        index = (index + 1) & (capacity - 1);
    }
    slots[index] = slot;
}

[[nodiscard]]
static bool ensure_accum_capacity(beamsearch_decoder_t *decoder,
                                  size_t required_entries) {
    size_t capacity = 0;
    if (!hash_capacity_for(required_entries, &capacity)) {
        return false;
    }
    if (capacity <= decoder->accum_capacity) {
        return true;
    }

    beamsearch_accum_slot_t *slots = (beamsearch_accum_slot_t *)calloc_array(
        capacity, sizeof(beamsearch_accum_slot_t));
    if (slots == nullptr) {
        return false;
    }

    for (size_t i = 0; i < decoder->accum_capacity; i += 1) {
        if (decoder->accum_slots[i].epoch == decoder->accum_epoch) {
            accum_insert_existing(slots, capacity, decoder->accum_slots[i]);
        }
    }

    free(decoder->accum_slots);
    decoder->accum_slots = slots;
    decoder->accum_capacity = capacity;
    return true;
}

static void reset_accumulator(beamsearch_decoder_t *decoder) {
    decoder->accum_count = 0;
    if (decoder->accum_epoch == UINT32_MAX) {
        memset(decoder->accum_slots, 0,
               decoder->accum_capacity * sizeof(decoder->accum_slots[0]));
        decoder->accum_epoch = BEAMSEARCH_INITIAL_EPOCH;
        return;
    }
    decoder->accum_epoch += 1;
}

[[nodiscard]]
static beamsearch_accum_slot_t *find_or_add_accum_slot(
    beamsearch_decoder_t *decoder,
    size_t node) {
    auto index = hash_node(node) & (decoder->accum_capacity - 1);
    while (decoder->accum_slots[index].epoch == decoder->accum_epoch) {
        if (decoder->accum_slots[index].node == node) {
            return &decoder->accum_slots[index];
        }
        index = (index + 1) & (decoder->accum_capacity - 1);
    }

    decoder->accum_slots[index].node = node;
    decoder->accum_slots[index].p_blank = BEAMSEARCH_LOG_ZERO;
    decoder->accum_slots[index].p_nonblank = BEAMSEARCH_LOG_ZERO;
    decoder->accum_slots[index].epoch = decoder->accum_epoch;
    decoder->accum_count += 1;
    return &decoder->accum_slots[index];
}

static void accum_add_blank(beamsearch_decoder_t *decoder, size_t node,
                            float value) {
    auto slot = find_or_add_accum_slot(decoder, node);
    slot->p_blank = logaddexp_pair(slot->p_blank, value);
}

static void accum_add_nonblank(beamsearch_decoder_t *decoder, size_t node,
                               float value) {
    auto slot = find_or_add_accum_slot(decoder, node);
    slot->p_nonblank = logaddexp_pair(slot->p_nonblank, value);
}

static void reset_trie(beamsearch_decoder_t *decoder) {
    if (decoder->child_capacity != 0) {
        memset(decoder->child_slots, 0,
               decoder->child_capacity * sizeof(decoder->child_slots[0]));
    }

    decoder->child_count = 0;
    decoder->node_count = 1;
    decoder->nodes[0].parent = BEAMSEARCH_INVALID_NODE;
    decoder->nodes[0].token = -1;
    decoder->nodes[0].length = 0;
}

[[nodiscard]]
static bool get_or_create_child(beamsearch_decoder_t *decoder,
                                size_t parent,
                                int32_t token,
                                size_t *child_out) {
    size_t required_child_count = 0;
    if (!checked_add_size(decoder->child_count, 1, &required_child_count)) {
        return false;
    }
    if (!ensure_child_capacity(decoder, required_child_count)) {
        return false;
    }
    size_t required_node_count = 0;
    if (!checked_add_size(decoder->node_count, 1, &required_node_count)) {
        return false;
    }
    if (!ensure_node_capacity(decoder, required_node_count)) {
        return false;
    }

    auto index = hash_child(parent, token) & (decoder->child_capacity - 1);
    while (decoder->child_slots[index].occupied) {
        if (decoder->child_slots[index].parent == parent &&
            decoder->child_slots[index].token == token) {
            *child_out = decoder->child_slots[index].child;
            return true;
        }
        index = (index + 1) & (decoder->child_capacity - 1);
    }

    auto child = decoder->node_count;
    decoder->node_count += 1;
    decoder->nodes[child].parent = parent;
    decoder->nodes[child].token = token;
    decoder->nodes[child].length = decoder->nodes[parent].length + 1;

    decoder->child_slots[index].parent = parent;
    decoder->child_slots[index].token = token;
    decoder->child_slots[index].child = child;
    decoder->child_slots[index].occupied = true;
    decoder->child_count += 1;

    *child_out = child;
    return true;
}

[[nodiscard]]
static size_t nonblank_token_count(size_t vocab_size) {
    if (vocab_size == 0) {
        return 0;
    }
    return vocab_size - 1;
}

[[nodiscard]]
static bool token_score_better(float lhs_score,
                               int32_t lhs_token,
                               float rhs_score,
                               int32_t rhs_token) {
    if (lhs_score > rhs_score) {
        return true;
    }
    if (lhs_score < rhs_score) {
        return false;
    }
    return lhs_token < rhs_token;
}

static void insert_top_token_arrays(int32_t *tokens,
                                    float *scores,
                                    size_t *selected_count,
                                    size_t cutoff,
                                    int32_t token,
                                    float score) {
    if (*selected_count == cutoff &&
        !token_score_better(score, token, scores[cutoff - 1],
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
           token_score_better(score, token, scores[insert_at - 1],
                              tokens[insert_at - 1])) {
        scores[insert_at] = scores[insert_at - 1];
        tokens[insert_at] = tokens[insert_at - 1];
        insert_at -= 1;
    }

    scores[insert_at] = score;
    tokens[insert_at] = token;
}

static void insert_top_token(beamsearch_decoder_t *decoder,
                             size_t *selected_count,
                             size_t cutoff,
                             int32_t token,
                             float score) {
    insert_top_token_arrays(decoder->selected_tokens, decoder->selected_scores,
                            selected_count, cutoff, token, score);
}

static void select_top_tokens_range(int32_t *tokens,
                                    float *scores,
                                    size_t *selected_count,
                                    size_t cutoff,
                                    const float *frame,
                                    size_t begin,
                                    size_t end,
                                    int32_t blank_id) {
    constexpr size_t lanes = 8;
    auto token = begin;
    while (end - token >= lanes) {
        auto chunk = psimd_loadu_f32x8(frame + token);
        auto chunk_best = psimd_reduce_max_f32x8(chunk);
        if (*selected_count < cutoff || chunk_best > scores[cutoff - 1]) {
            for (size_t lane = 0; lane < lanes; lane += 1) {
                auto token_index = token + lane;
                auto token_id = (int32_t)token_index;
                if (token_id != blank_id) {
                    insert_top_token_arrays(tokens, scores, selected_count,
                                            cutoff, token_id,
                                            frame[token_index]);
                }
            }
        }
        token += lanes;
    }

    while (token < end) {
        auto token_id = (int32_t)token;
        if (token_id != blank_id) {
            insert_top_token_arrays(tokens, scores, selected_count, cutoff,
                                    token_id, frame[token]);
        }
        token += 1;
    }
}

[[nodiscard]]
static bool select_top_tokens_threaded(beamsearch_decoder_t *decoder,
                                       const float *frame,
                                       size_t vocab_size,
                                       size_t cutoff,
                                       size_t *selected_count_out);

[[nodiscard]]
static bool select_top_tokens(beamsearch_decoder_t *decoder,
                              const float *frame,
                              size_t vocab_size,
                              size_t *selected_count_out) {
    auto available = nonblank_token_count(vocab_size);
    auto cutoff = decoder->config.token_cutoff;
    if (cutoff > available) {
        cutoff = available;
    }

    *selected_count_out = 0;
    if (cutoff == 0) {
        return true;
    }

    if (!ensure_selected_token_capacity(decoder, cutoff) ||
        !ensure_selected_score_capacity(decoder, cutoff)) {
        return false;
    }

    if (decoder->config.token_cutoff >= available) {
        for (size_t token = 0; token < vocab_size; token += 1) {
            if ((int32_t)token != decoder->config.blank_id) {
                decoder->selected_tokens[*selected_count_out] = (int32_t)token;
                *selected_count_out += 1;
            }
        }
        return true;
    }

#ifdef BEAMSEARCH_ENABLE_CUDA
    if (vocab_size >= BEAMSEARCH_MIN_CUDA_SELECT_TOKENS &&
        ensure_cuda_context(decoder, 1, vocab_size, cutoff) &&
        beamsearch_cuda_select_top_tokens(
            decoder->cuda_context, frame, vocab_size, cutoff,
            decoder->config.blank_id, decoder->selected_tokens,
            decoder->selected_scores, selected_count_out)) {
        return true;
    }
#endif

    if (decoder->thread_count > 1 &&
        vocab_size >= BEAMSEARCH_MIN_THREADED_SELECT_TOKENS) {
        return select_top_tokens_threaded(decoder, frame, vocab_size, cutoff,
                                          selected_count_out);
    }

    select_top_tokens_range(decoder->selected_tokens, decoder->selected_scores,
                            selected_count_out, cutoff, frame, 0, vocab_size,
                            decoder->config.blank_id);

    return true;
}

[[nodiscard]]
static bool prepare_frame_storage(beamsearch_decoder_t *decoder,
                                  size_t selected_count) {
    size_t selected_with_blank = 0;
    if (!checked_add_size(selected_count, 1, &selected_with_blank)) {
        return false;
    }

    size_t expansion_count = 0;
    if (!checked_mul_size(decoder->current_count, selected_with_blank,
                          &expansion_count)) {
        return false;
    }
    size_t required_accum_count = 0;
    if (!checked_add_size(expansion_count, 1, &required_accum_count)) {
        return false;
    }
    if (!ensure_accum_capacity(decoder, required_accum_count)) {
        return false;
    }

    size_t new_child_count = 0;
    if (!checked_mul_size(decoder->current_count, selected_count,
                          &new_child_count)) {
        return false;
    }
    size_t required_child_count = 0;
    if (!checked_add_size(decoder->child_count, new_child_count,
                          &required_child_count)) {
        return false;
    }
    if (!ensure_child_capacity(decoder, required_child_count)) {
        return false;
    }
    size_t required_node_count = 0;
    if (!checked_add_size(decoder->node_count, new_child_count,
                          &required_node_count)) {
        return false;
    }
    if (!ensure_node_capacity(decoder, required_node_count)) {
        return false;
    }

    return true;
}

static void prune_accumulator_to_next(beamsearch_decoder_t *decoder) {
    decoder->next_count = 0;
    for (size_t i = 0; i < decoder->accum_capacity; i += 1) {
        if (decoder->accum_slots[i].epoch != decoder->accum_epoch) {
            continue;
        }

        beamsearch_beam_t candidate = {
            .node = decoder->accum_slots[i].node,
            .p_blank = decoder->accum_slots[i].p_blank,
            .p_nonblank = decoder->accum_slots[i].p_nonblank,
            .score = logaddexp_pair(decoder->accum_slots[i].p_blank,
                                    decoder->accum_slots[i].p_nonblank),
        };
        heap_consider(decoder->next_beams, &decoder->next_count,
                      decoder->next_capacity, candidate);
    }

    sort_beams_desc(decoder->next_beams, decoder->next_count);

    auto tmp_beams = decoder->current_beams;
    decoder->current_beams = decoder->next_beams;
    decoder->next_beams = tmp_beams;

    auto tmp_capacity = decoder->current_capacity;
    decoder->current_capacity = decoder->next_capacity;
    decoder->next_capacity = tmp_capacity;

    decoder->current_count = decoder->next_count;
    decoder->next_count = 0;
}

[[nodiscard]]
static bool expand_frame(beamsearch_decoder_t *decoder,
                         const float *frame,
                         size_t selected_count) {
    reset_accumulator(decoder);

    auto blank_score = frame[decoder->config.blank_id];
    for (size_t i = 0; i < decoder->current_count; i += 1) {
        const beamsearch_beam_t *beam = &decoder->current_beams[i];
        auto total = beam_total_score(beam);
        accum_add_blank(decoder, beam->node, total + blank_score);

        auto last_token = decoder->nodes[beam->node].token;
        for (size_t j = 0; j < selected_count; j += 1) {
            auto token = decoder->selected_tokens[j];
            auto token_score = frame[token];

            if (token == last_token) {
                accum_add_nonblank(decoder, beam->node,
                                   beam->p_nonblank + token_score);

                size_t child = 0;
                if (!get_or_create_child(decoder, beam->node, token, &child)) {
                    return false;
                }
                accum_add_nonblank(decoder, child,
                                   beam->p_blank + token_score);
                continue;
            }

            size_t child = 0;
            if (!get_or_create_child(decoder, beam->node, token, &child)) {
                return false;
            }
            accum_add_nonblank(decoder, child, total + token_score);
        }
    }

    prune_accumulator_to_next(decoder);
    return true;
}

[[nodiscard]]
static bool worker_append_transition(beamsearch_worker_t *worker,
                                     size_t node,
                                     float value,
                                     bool nonblank) {
    if (worker->transition_count >= worker->transition_capacity) {
        return false;
    }

    worker->transitions[worker->transition_count].node = node;
    worker->transitions[worker->transition_count].value = value;
    worker->transitions[worker->transition_count].nonblank = nonblank;
    worker->transition_count += 1;
    return true;
}

static void expand_worker_compute(beamsearch_worker_t *worker) {
    auto decoder = worker->decoder;
    worker->transition_count = 0;
    worker->ok = true;

    auto blank_score = worker->frame[decoder->config.blank_id];
    for (size_t i = worker->beam_begin; i < worker->beam_end; i += 1) {
        const beamsearch_beam_t *beam = &decoder->current_beams[i];
        auto total = beam_total_score(beam);
        if (!worker_append_transition(worker, beam->node,
                                      total + blank_score, false)) {
            worker->ok = false;
            return;
        }

        auto last_token = decoder->nodes[beam->node].token;
        for (size_t j = 0; j < worker->selected_count; j += 1) {
            auto token = decoder->selected_tokens[j];
            auto token_score = worker->frame[token];
            auto child = decoder->expanded_children[
                i * worker->selected_count + j];

            if (token == last_token) {
                if (!worker_append_transition(worker, beam->node,
                                              beam->p_nonblank + token_score,
                                              true) ||
                    !worker_append_transition(worker, child,
                                              beam->p_blank + token_score,
                                              true)) {
                    worker->ok = false;
                    return;
                }
                continue;
            }

            if (!worker_append_transition(worker, child, total + token_score,
                                          true)) {
                worker->ok = false;
                return;
            }
        }
    }
}

static void select_worker_compute(beamsearch_worker_t *worker) {
    auto decoder = worker->decoder;
    worker->local_count = 0;
    worker->ok = true;

    select_top_tokens_range(worker->local_tokens, worker->local_scores,
                            &worker->local_count, worker->selected_count,
                            worker->frame, worker->vocab_begin,
                            worker->vocab_end, decoder->config.blank_id);
}

static void *expand_worker_main(void *arg) {
    auto worker = (beamsearch_worker_t *)arg;
    auto decoder = worker->decoder;

    if (pthread_mutex_lock(&decoder->worker_mutex) != 0) {
        return nullptr;
    }

    while (true) {
        while (!decoder->worker_stop &&
               worker->seen_generation == decoder->worker_generation) {
            (void)pthread_cond_wait(&decoder->worker_start,
                                    &decoder->worker_mutex);
        }
        if (decoder->worker_stop) {
            break;
        }

        worker->seen_generation = decoder->worker_generation;
        auto active = worker->worker_index < decoder->active_worker_count;

        if (pthread_mutex_unlock(&decoder->worker_mutex) != 0) {
            return nullptr;
        }
        if (active) {
            if (decoder->worker_task == beamsearch_worker_task_select) {
                select_worker_compute(worker);
            } else {
                expand_worker_compute(worker);
            }
        }
        if (pthread_mutex_lock(&decoder->worker_mutex) != 0) {
            return nullptr;
        }

        if (active) {
            decoder->workers_done += 1;
            if (decoder->workers_done == decoder->active_worker_count) {
                (void)pthread_cond_signal(&decoder->worker_done);
            }
        }
    }

    (void)pthread_mutex_unlock(&decoder->worker_mutex);
    return nullptr;
}

[[nodiscard]]
static bool precompute_expanded_children(beamsearch_decoder_t *decoder,
                                         size_t selected_count) {
    size_t child_count = 0;
    if (!checked_mul_size(decoder->current_count, selected_count,
                          &child_count)) {
        return false;
    }
    if (!ensure_expanded_child_capacity(decoder, child_count)) {
        return false;
    }

    for (size_t i = 0; i < decoder->current_count; i += 1) {
        auto parent = decoder->current_beams[i].node;
        for (size_t j = 0; j < selected_count; j += 1) {
            size_t child = 0;
            if (!get_or_create_child(decoder, parent,
                                     decoder->selected_tokens[j], &child)) {
                return false;
            }
            decoder->expanded_children[i * selected_count + j] = child;
        }
    }

    return true;
}

[[nodiscard]]
static size_t effective_thread_count(const beamsearch_decoder_t *decoder,
                                     size_t selected_count) {
    if (decoder->thread_count < 2 || decoder->current_count < 2 ||
        selected_count == 0) {
        return 1;
    }

    size_t expansion_work = 0;
    if (!checked_mul_size(decoder->current_count, selected_count,
                          &expansion_work) ||
        expansion_work < BEAMSEARCH_MIN_THREADED_EXPANSIONS) {
        return 1;
    }

    auto thread_count = decoder->thread_count;
    if (thread_count > decoder->current_count) {
        thread_count = decoder->current_count;
    }
    return thread_count;
}

[[nodiscard]]
static bool prepare_workers(beamsearch_decoder_t *decoder,
                            const float *frame,
                            size_t selected_count,
                            size_t thread_count) {
    if (thread_count > decoder->worker_capacity) {
        return false;
    }

    size_t per_beam_transitions = 0;
    size_t doubled_selected = 0;
    if (!checked_mul_size(selected_count, 2, &doubled_selected) ||
        !checked_add_size(doubled_selected, 1, &per_beam_transitions)) {
        return false;
    }

    for (size_t i = 0; i < thread_count; i += 1) {
        size_t begin = 0;
        size_t end = 0;
        partition_range(decoder->current_count, thread_count, i, &begin, &end);
        size_t worker_capacity = 0;
        if (!checked_mul_size(end - begin, per_beam_transitions,
                              &worker_capacity)) {
            return false;
        }
        if (!ensure_worker_transition_capacity(&decoder->workers[i],
                                               worker_capacity)) {
            return false;
        }

        decoder->workers[i].decoder = decoder;
        decoder->workers[i].frame = frame;
        decoder->workers[i].beam_begin = begin;
        decoder->workers[i].beam_end = end;
        decoder->workers[i].selected_count = selected_count;
        decoder->workers[i].transition_count = 0;
        decoder->workers[i].ok = true;
    }

    return true;
}

[[nodiscard]]
static bool prepare_select_workers(beamsearch_decoder_t *decoder,
                                   const float *frame,
                                   size_t vocab_size,
                                   size_t cutoff,
                                   size_t thread_count) {
    if (thread_count > decoder->worker_capacity) {
        return false;
    }

    for (size_t i = 0; i < thread_count; i += 1) {
        if (!ensure_worker_select_capacity(&decoder->workers[i], cutoff)) {
            return false;
        }

        size_t begin = 0;
        size_t end = 0;
        partition_range(vocab_size, thread_count, i, &begin, &end);
        decoder->workers[i].decoder = decoder;
        decoder->workers[i].frame = frame;
        decoder->workers[i].vocab_begin = begin;
        decoder->workers[i].vocab_end = end;
        decoder->workers[i].selected_count = cutoff;
        decoder->workers[i].local_count = 0;
        decoder->workers[i].ok = true;
    }

    return true;
}

[[nodiscard]]
static bool init_worker_sync(beamsearch_decoder_t *decoder) {
    if (decoder->worker_sync_ready) {
        return true;
    }

    if (pthread_mutex_init(&decoder->worker_mutex, nullptr) != 0) {
        return false;
    }
    if (pthread_cond_init(&decoder->worker_start, nullptr) != 0) {
        (void)pthread_mutex_destroy(&decoder->worker_mutex);
        return false;
    }
    if (pthread_cond_init(&decoder->worker_done, nullptr) != 0) {
        (void)pthread_cond_destroy(&decoder->worker_start);
        (void)pthread_mutex_destroy(&decoder->worker_mutex);
        return false;
    }

    decoder->worker_sync_ready = true;
    return true;
}

static void stop_worker_pool(beamsearch_decoder_t *decoder) {
    if (decoder->workers_started) {
        if (pthread_mutex_lock(&decoder->worker_mutex) == 0) {
            decoder->worker_stop = true;
            decoder->worker_generation += 1;
            (void)pthread_cond_broadcast(&decoder->worker_start);
            (void)pthread_mutex_unlock(&decoder->worker_mutex);
        }

        for (size_t i = 0; i < decoder->worker_count; i += 1) {
            (void)pthread_join(decoder->workers[i].thread, nullptr);
        }
        decoder->workers_started = false;
        decoder->worker_count = 0;
    }

    if (decoder->worker_sync_ready) {
        (void)pthread_cond_destroy(&decoder->worker_done);
        (void)pthread_cond_destroy(&decoder->worker_start);
        (void)pthread_mutex_destroy(&decoder->worker_mutex);
        decoder->worker_sync_ready = false;
    }
}

[[nodiscard]]
static bool start_worker_pool(beamsearch_decoder_t *decoder,
                              size_t requested_thread_count) {
    if (requested_thread_count < 2) {
        return true;
    }
    if (requested_thread_count > decoder->thread_count) {
        requested_thread_count = decoder->thread_count;
    }
    if (requested_thread_count < 2) {
        return true;
    }
    if (decoder->workers_started &&
        decoder->worker_count >= requested_thread_count) {
        return true;
    }
    if (decoder->workers_started) {
        stop_worker_pool(decoder);
    }

    if (!ensure_worker_capacity(decoder, requested_thread_count) ||
        !init_worker_sync(decoder)) {
        return false;
    }

    decoder->worker_stop = false;
    decoder->worker_count = requested_thread_count;
    decoder->active_worker_count = 0;
    decoder->workers_done = 0;
    decoder->worker_generation = 0;

    size_t started = 0;
    for (size_t i = 0; i < requested_thread_count; i += 1) {
        decoder->workers[i].decoder = decoder;
        decoder->workers[i].worker_index = i;
        decoder->workers[i].seen_generation = 0;
        decoder->workers[i].transition_count = 0;
        decoder->workers[i].ok = true;

        if (pthread_create(&decoder->workers[i].thread, nullptr,
                           expand_worker_main, &decoder->workers[i]) != 0) {
            if (pthread_mutex_lock(&decoder->worker_mutex) == 0) {
                decoder->worker_stop = true;
                decoder->worker_generation += 1;
                (void)pthread_cond_broadcast(&decoder->worker_start);
                (void)pthread_mutex_unlock(&decoder->worker_mutex);
            }
            for (size_t j = 0; j < started; j += 1) {
                (void)pthread_join(decoder->workers[j].thread, nullptr);
            }
            decoder->worker_count = 0;
            return false;
        }
        started += 1;
    }

    decoder->workers_started = true;
    return true;
}

[[nodiscard]]
static bool run_worker_task(beamsearch_decoder_t *decoder,
                            beamsearch_worker_task_t task,
                            size_t active_worker_count) {
    if (pthread_mutex_lock(&decoder->worker_mutex) != 0) {
        return false;
    }

    decoder->worker_task = task;
    decoder->active_worker_count = active_worker_count;
    decoder->workers_done = 0;
    decoder->worker_generation += 1;
    if (pthread_cond_broadcast(&decoder->worker_start) != 0) {
        (void)pthread_mutex_unlock(&decoder->worker_mutex);
        return false;
    }
    while (decoder->workers_done < decoder->active_worker_count) {
        if (pthread_cond_wait(&decoder->worker_done,
                              &decoder->worker_mutex) != 0) {
            (void)pthread_mutex_unlock(&decoder->worker_mutex);
            return false;
        }
    }

    if (pthread_mutex_unlock(&decoder->worker_mutex) != 0) {
        return false;
    }

    for (size_t i = 0; i < active_worker_count; i += 1) {
        if (!decoder->workers[i].ok) {
            return false;
        }
    }
    return true;
}

[[nodiscard]]
static bool select_top_tokens_threaded(beamsearch_decoder_t *decoder,
                                       const float *frame,
                                       size_t vocab_size,
                                       size_t cutoff,
                                       size_t *selected_count_out) {
    auto thread_count = decoder->thread_count;
    if (thread_count > vocab_size) {
        thread_count = vocab_size;
    }
    if (thread_count < 2) {
        select_top_tokens_range(decoder->selected_tokens,
                                decoder->selected_scores, selected_count_out,
                                cutoff, frame, 0, vocab_size,
                                decoder->config.blank_id);
        return true;
    }

    if (!start_worker_pool(decoder, thread_count) ||
        !prepare_select_workers(decoder, frame, vocab_size, cutoff,
                                thread_count) ||
        !run_worker_task(decoder, beamsearch_worker_task_select,
                         thread_count)) {
        return false;
    }

    *selected_count_out = 0;
    for (size_t worker_index = 0; worker_index < thread_count;
         worker_index += 1) {
        const beamsearch_worker_t *worker = &decoder->workers[worker_index];
        for (size_t i = 0; i < worker->local_count; i += 1) {
            insert_top_token(decoder, selected_count_out, cutoff,
                             worker->local_tokens[i], worker->local_scores[i]);
        }
    }

    return true;
}

static void merge_worker_transitions(beamsearch_decoder_t *decoder,
                                     const beamsearch_worker_t *worker) {
    for (size_t i = 0; i < worker->transition_count; i += 1) {
        const beamsearch_transition_t *transition = &worker->transitions[i];
        if (transition->nonblank) {
            accum_add_nonblank(decoder, transition->node, transition->value);
        } else {
            accum_add_blank(decoder, transition->node, transition->value);
        }
    }
}

[[nodiscard]]
static bool expand_frame_threaded(beamsearch_decoder_t *decoder,
                                  const float *frame,
                                  size_t selected_count) {
    auto thread_count = effective_thread_count(decoder, selected_count);
    if (thread_count < 2) {
        return expand_frame(decoder, frame, selected_count);
    }

    if (!precompute_expanded_children(decoder, selected_count) ||
        !start_worker_pool(decoder, thread_count)) {
        return false;
    }

    reset_accumulator(decoder);

    if (!prepare_workers(decoder, frame, selected_count, thread_count)) {
        return false;
    }
    if (!run_worker_task(decoder, beamsearch_worker_task_expand,
                         thread_count)) {
        return false;
    }

    for (size_t i = 0; i < thread_count; i += 1) {
        merge_worker_transitions(decoder, &decoder->workers[i]);
    }

    prune_accumulator_to_next(decoder);
    return true;
}

static void reconstruct_tokens(const beamsearch_decoder_t *decoder,
                               size_t node,
                               int32_t *output_tokens,
                               size_t token_count) {
    auto index = token_count;
    while (node != BEAMSEARCH_INVALID_NODE && index > 0) {
        index -= 1;
        output_tokens[index] = decoder->nodes[node].token;
        node = decoder->nodes[node].parent;
    }
}

beamsearch_decoder_t *beamsearch_decoder_create(
    const beamsearch_config_t *config) {
    if (config == nullptr || config->beam_width == 0 ||
        config->token_cutoff == 0 || config->blank_id < 0) {
        return nullptr;
    }

    beamsearch_decoder_t *decoder = (beamsearch_decoder_t *)calloc_array(
        1, sizeof(beamsearch_decoder_t));
    if (decoder == nullptr) {
        return nullptr;
    }

    decoder->config = *config;
    decoder->config.thread_count =
        sanitize_thread_count(decoder->config.thread_count);
    decoder->thread_count = decoder->config.thread_count;
    decoder->accum_epoch = BEAMSEARCH_INITIAL_EPOCH;

    if (!ensure_node_capacity(decoder, 1) ||
        !ensure_child_capacity(decoder, BEAMSEARCH_MIN_HASH_CAPACITY / 2) ||
        !ensure_accum_capacity(decoder, BEAMSEARCH_MIN_HASH_CAPACITY / 2) ||
        !ensure_beam_capacity(&decoder->current_beams,
                              &decoder->current_capacity, 0,
                              config->beam_width) ||
        !ensure_beam_capacity(&decoder->next_beams, &decoder->next_capacity, 0,
                              config->beam_width)) {
        beamsearch_decoder_destroy(decoder);
        return nullptr;
    }

    return decoder;
}

void beamsearch_decoder_destroy(beamsearch_decoder_t *decoder) {
    if (decoder == nullptr) {
        return;
    }

    stop_worker_pool(decoder);

    free(decoder->nodes);
    free(decoder->child_slots);
    free(decoder->accum_slots);
    free(decoder->current_beams);
    free(decoder->next_beams);
    free(decoder->selected_tokens);
    free(decoder->selected_scores);
    free(decoder->expanded_children);
#ifdef BEAMSEARCH_ENABLE_CUDA
    beamsearch_cuda_context_destroy(decoder->cuda_context);
    free(decoder->cuda_selected_tokens);
    free(decoder->cuda_selected_counts);
#endif
    for (size_t i = 0; i < decoder->worker_capacity; i += 1) {
        free(decoder->workers[i].local_tokens);
        free(decoder->workers[i].local_scores);
        free(decoder->workers[i].transitions);
    }
    free(decoder->workers);
    free(decoder);
}

beamsearch_status_t beamsearch_decoder_decode(
    beamsearch_decoder_t *decoder,
    const float *log_probs,
    size_t frame_count,
    size_t vocab_size,
    int32_t *output_tokens,
    size_t output_capacity,
    beamsearch_result_t *result) {
    if (decoder == nullptr || result == nullptr || vocab_size == 0 ||
        vocab_size > (size_t)INT32_MAX + 1 ||
        decoder->config.blank_id < 0 ||
        (size_t)decoder->config.blank_id >= vocab_size ||
        (frame_count != 0 && vocab_size > SIZE_MAX / frame_count) ||
        (frame_count > 0 && log_probs == nullptr) ||
        (output_capacity > 0 && output_tokens == nullptr)) {
        return beamsearch_status_invalid_argument;
    }

    result->token_count = 0;
    result->log_score = BEAMSEARCH_LOG_ZERO;

    if (frame_count == 0) {
        result->log_score = 0.0f;
        return beamsearch_status_ok;
    }

    if (!ensure_node_capacity(decoder, 1)) {
        return beamsearch_status_out_of_memory;
    }
    reset_trie(decoder);

    decoder->current_count = 1;
    decoder->current_beams[0].node = 0;
    decoder->current_beams[0].p_blank = 0.0f;
    decoder->current_beams[0].p_nonblank = BEAMSEARCH_LOG_ZERO;
    decoder->current_beams[0].score = 0.0f;
    decoder->next_count = 0;

#ifdef BEAMSEARCH_ENABLE_CUDA
    bool use_cuda_preselected = false;
    auto cuda_cutoff = decoder->config.token_cutoff;
    auto available_tokens = nonblank_token_count(vocab_size);
    if (cuda_cutoff > available_tokens) {
        cuda_cutoff = available_tokens;
    }
    if (vocab_size >= BEAMSEARCH_MIN_CUDA_SELECT_TOKENS &&
        cuda_cutoff != 0 &&
        decoder->config.token_cutoff < available_tokens &&
        ensure_cuda_context(decoder, frame_count, vocab_size, cuda_cutoff) &&
        ensure_cuda_selected_capacity(decoder, frame_count, cuda_cutoff) &&
        beamsearch_cuda_select_top_tokens_batch(
            decoder->cuda_context, log_probs, frame_count, vocab_size,
            cuda_cutoff, decoder->config.blank_id,
            decoder->cuda_selected_tokens, decoder->cuda_selected_counts)) {
        use_cuda_preselected = true;
    }
#endif

    for (size_t frame_index = 0; frame_index < frame_count; frame_index += 1) {
        auto frame = log_probs + frame_index * vocab_size;
        size_t selected_count = 0;

#ifdef BEAMSEARCH_ENABLE_CUDA
        if (use_cuda_preselected) {
            selected_count = decoder->cuda_selected_counts[frame_index];
            if (selected_count > cuda_cutoff ||
                !ensure_selected_token_capacity(decoder, selected_count)) {
                return beamsearch_status_out_of_memory;
            }
            if (selected_count != 0) {
                memcpy(decoder->selected_tokens,
                       decoder->cuda_selected_tokens +
                           frame_index * cuda_cutoff,
                       selected_count * sizeof(decoder->selected_tokens[0]));
            }
        } else
#endif
        if (!select_top_tokens(decoder, frame, vocab_size, &selected_count)) {
            return beamsearch_status_out_of_memory;
        }
        if (!prepare_frame_storage(decoder, selected_count)) {
            return beamsearch_status_out_of_memory;
        }
        if (!expand_frame_threaded(decoder, frame, selected_count)) {
            return beamsearch_status_out_of_memory;
        }
    }

    if (decoder->current_count == 0) {
        result->log_score = BEAMSEARCH_LOG_ZERO;
        return beamsearch_status_ok;
    }

    const beamsearch_beam_t *best = &decoder->current_beams[0];
    auto required_tokens = decoder->nodes[best->node].length;
    result->token_count = required_tokens;
    result->log_score = best->score;

    if (required_tokens > output_capacity) {
        return beamsearch_status_output_too_small;
    }
    if (required_tokens != 0) {
        reconstruct_tokens(decoder, best->node, output_tokens, required_tokens);
    }

    return beamsearch_status_ok;
}
