#include "Pipeline.h"
#include "common.h"
#include "utils.h"

// Static sampling methods
void Pipeline::sample_repetition_penalty(token_data_array* candidates, const int* last_tokens, size_t last_tokens_size,
                               float penalty) {
    if (last_tokens_size == 0 || penalty == 1.0f) {
        return;
    }

    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = std::find(last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
        if (token_iter == last_tokens + last_tokens_size) {
            continue;
        }

        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty;
        } else {
            candidates->data[i].logit /= penalty;
        }
    }

    candidates->sorted = false;
}

void Pipeline::sample_frequency_and_presence_penalties(token_data_array* candidates, const int* last_tokens_p,
                                             size_t last_tokens_size, float alpha_frequency, float alpha_presence) {
    if (last_tokens_size == 0 || (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
        return;
    }

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<int, int> token_count;
    for (size_t i = 0; i < last_tokens_size; ++i) {
        token_count[last_tokens_p[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        int count = token_iter->second;
        candidates->data[i].logit -= float(count) * alpha_frequency + float(count > 0) * alpha_presence;
    }

    candidates->sorted = false;
}

int Pipeline::sample_token_greedy(token_data_array* candidates) {
    // Find max element
    auto max_iter =
        std::max_element(candidates->data, candidates->data + candidates->size,
                         [](const token_data& a, const token_data& b) { return a.logit < b.logit; });

    int result = max_iter->id;
    return result;
}

void Pipeline::sample_temperature(token_data_array* candidates_p, float temp) {
    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= temp;
    }
}

void Pipeline::sample_softmax(token_data_array* candidates) {
    assert(candidates->size > 0);

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size,
                  [](const token_data& a, const token_data& b) { return a.logit > b.logit; });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }
}

int Pipeline::sample_token(token_data_array* candidates) {
    sample_softmax(candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    auto& rng = random_generator;
    int idx = dist(rng);

    int result = candidates->data[idx].id;
    return result;
}

void Pipeline::sample_top_k(token_data_array* candidates, int k, size_t min_keep) {
    k = std::max(k, (int)min_keep);
    k = std::min(k, (int)candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const token_data& a, const token_data& b) { return a.logit > b.logit; };
        if (k == (int)candidates->size) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
        candidates->sorted = true;
    }

    candidates->size = k;
}

int Pipeline::sample_token_mirostat(const int n_vocab, token_data_array* candidates, float tau, float eta, int m, float* mu) {
    auto N = float(n_vocab);

    sample_softmax(candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    sample_top_k(candidates, int(k), 1);
    int X = sample_token(candidates);

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data,
                                 std::find_if(candidates->data, candidates->data + candidates->size,
                                              [&](const token_data& candidate) { return candidate.id == X; }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    return X;
}

int Pipeline::sample_token_mirostat_v2(token_data_array* candidates, float tau, float eta, float* mu) {
    sample_softmax(candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(
        candidates->data, std::find_if(candidates->data, candidates->data + candidates->size,
                                       [&](const token_data& candidate) { return -log2f(candidate.p) > *mu; }));

    // Normalize the probabilities of the remaining words
    sample_softmax(candidates);

    // Sample the next word X from the remaining words
    int X = sample_token(candidates);

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data,
                                 std::find_if(candidates->data, candidates->data + candidates->size,
                                              [&](const token_data& candidate) { return candidate.id == X; }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    return X;
}

void Pipeline::sample_tail_free(token_data_array* candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    sample_softmax(candidates);

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);
    for (float& value : second_derivatives) {
        value /= second_derivatives_sum;
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;
}

void Pipeline::sample_typical(token_data_array* candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    sample_softmax(candidates);

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return shifted_scores[a] < shifted_scores[b]; });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();
}

void Pipeline::sample_top_p(token_data_array* candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    sample_softmax(candidates);

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is greater than p or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;
}

// Encapsulate the sampling process
int Pipeline::sample_next_token(std::vector<float>& logits, std::vector<int>& last_n_tokens, const qwen_params& generation_config) {
    const int32_t max_context_length = generation_config.n_ctx;
    const int vocab_size = generation_config.n_vocab;
    std::vector<token_data> candidates;
    candidates.reserve(vocab_size);
    for (int token_id = 0; token_id < vocab_size; token_id++) {
        candidates.emplace_back(token_data{token_id, logits[token_id], 0.0f});
    }
    token_data_array candidiate_p = {candidates.data(), candidates.size(), false};
    const int32_t repeat_last_n = generation_config.repeat_last_n < 0 ? max_context_length : generation_config.repeat_last_n;
    int32_t last_n_repeat = std::min(
        std::min((int)last_n_tokens.size(), repeat_last_n),
        max_context_length
    );
    sample_repetition_penalty(
        &candidiate_p,
        last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
        last_n_repeat,
        generation_config.repeat_penalty
    );
    sample_frequency_and_presence_penalties(
        &candidiate_p,
        last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
        last_n_repeat,
        generation_config.frequency_penalty,
        generation_config.presence_penalty
    );
    const float temperature = generation_config.temp;
    const int mirostat = generation_config.mirostat;
    int next_token_id = 0;
    if (temperature <= 0) {
        next_token_id = sample_token_greedy(&candidiate_p);
    } else {
        if (mirostat != 0) {
            // printf("\e[31m[INFO]\e[m Not implemented yet.");
        } else {
            sample_top_k(&candidiate_p, generation_config.top_k, 1);
            sample_tail_free(&candidiate_p, generation_config.tfs_z, 1);
            sample_typical(&candidiate_p, generation_config.typical_p, 1);
            sample_top_p(&candidiate_p, generation_config.top_p, 1);
            sample_temperature(&candidiate_p, temperature);
            next_token_id = sample_token(&candidiate_p);
        }
    }
    return next_token_id;
}

// Pipeline class implementation
Pipeline::Pipeline(void* model_ptr, const std::string& tiktoken_path, bool interactive, const qwen3_config& config)
    : tokenizer(tiktoken_path, config)
    , model(static_cast<Qwen3ForCausalLM*>(model_ptr))
    , config(config)
    , tiktoken_path(tiktoken_path)
    , interactive(interactive) {
}

Pipeline::~Pipeline() {
    // Note: We don't delete model here as it's owned by the caller
}

std::vector<int> Pipeline::encode(const std::string& text, int max_length) {
    return tokenizer.encode(text, max_length);
}

std::string Pipeline::decode(const std::vector<int>& tokens) {
    return tokenizer.decode(tokens);
}

// Helper for the main generation loop, takes already-encoded input_ids
std::vector<int> Pipeline::_generate_from_ids(std::vector<int> input_ids, const qwen_params& generation_config) {
    const int32_t max_context_length = generation_config.n_ctx;
    std::vector<int> last_n_tokens(max_context_length);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    std::vector<int> generate_ids;
    const int max_token_length = config.max_sqlen;
    const int vocab_size = generation_config.n_vocab;
    if (interactive) std::cout << "ASSISTANT: " << std::endl;
    int n_remain = generation_config.n_predict;
    int stop_generation_tolerance = 2;
    while (n_remain != 0 && stop_generation_tolerance) {
        std::vector<float> logits(vocab_size);
        int sqlen = input_ids.size();
        std::string PhaseName = (sqlen > 1) ? "[ P Stage ]" : "[ AG Stage ]";
        PROFILE_START(PhaseName);
        Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
        Qwen3ForCausalLM_Input model_input = {&input_ids_mat};
        Qwen3ForCausalLM_Output model_output = model->forward(model_input);
        Matrix3D<float> & last_token_logits = model_output.logits;
        memcpy(logits.data(), last_token_logits.data(), vocab_size*sizeof(float));
        PROFILE_START(PhaseName + ": Sampling");
        int next_token_id = sample_next_token(logits, last_n_tokens, generation_config);
        PROFILE_END(PhaseName + ": Sampling");
        PROFILE_END(PhaseName);
        if (next_token_id == config.eos_token_id) {
            // printf("\e[31m[INFO]\e[m EOS detected\n");
            stop_generation_tolerance--;
            continue;
        }
        if (next_token_id == config.pad_token_id) {
            // printf("\e[31m[INFO]\e[m padding token detected\n");
            continue;
        }
        stop_generation_tolerance = 2;
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(next_token_id);
        generate_ids.push_back(next_token_id);
        input_ids = std::vector<int>{next_token_id};
        if (interactive) {
            std::string output_text = decode(input_ids);
            std::cout << "\e[93m" << output_text << "\e[0m" << std::flush;
        }
        --n_remain;
    }
    if (interactive) std::cout << std::endl;
    Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();
    return generate_ids;
}

// New generate overload for chat history
std::vector<int> Pipeline::generate(const std::vector<std::string>& history, const qwen_params& generation_config) {
    const int max_token_length = config.max_sqlen;
    std::vector<int> input_ids = tokenizer.encode_history(history, max_token_length);
    return _generate_from_ids(input_ids, generation_config);
} 