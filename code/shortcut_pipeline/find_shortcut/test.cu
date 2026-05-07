#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <jansson.h>

namespace {

constexpr uint16_t kInvalidAnswerId = 0xFFFF;
constexpr uint16_t kNoKeywordId = 0;
constexpr int kMaxRuleKeywords = 4;
constexpr int kMaxQuestionTokens = 64;
constexpr int kMaxAnswerCandidates = 16;
constexpr int kCueMaskWords = 16;
constexpr int kThreadsPerBlock = 256;
constexpr int kDefaultBatchSize = 262144;

#define CUDA_CHECK(expr)                                                                          \
    do                                                                                            \
    {                                                                                             \
        cudaError_t err__ = (expr);                                                               \
        if (err__ != cudaSuccess)                                                                 \
        {                                                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

struct RawRule
{
    std::string rule_id;
    std::vector<std::string> text_keywords;
    std::vector<std::string> visual_cues;
    std::string answer;
    double confidence = 0.0;
    double support = 0.0;
};

struct CompactRule
{
    uint16_t answer_id = kInvalidAnswerId;
    uint16_t num_keywords = 0;
    uint16_t keyword_ids[kMaxRuleKeywords] = {0};
};

struct RuleRuntime
{
    std::string rule_id;
    CompactRule compact;
    uint16_t anchor_keyword_id = kNoKeywordId;
    std::vector<uint16_t> cue_ids;
};

struct EncodedQuestion
{
    int question_id = 0;
    int image_id = 0;
    uint16_t num_answer_candidates = 0;
    uint16_t answer_ids[kMaxAnswerCandidates] = {0};
    uint16_t num_tokens = 0;
    uint16_t token_ids[kMaxQuestionTokens] = {0};
};

struct AnswerCandidateSet
{
    uint16_t count = 0;
    uint16_t ids[kMaxAnswerCandidates] = {0};
};

struct CueMask
{
    uint64_t words[kCueMaskWords] = {0, 0};
};

struct AnswerCueSpan
{
    int group_offset = 0;
    int group_count = 0;
};

struct AnchorGroup
{
    uint16_t anchor_keyword_id = kNoKeywordId;
    uint16_t reserved = 0;
    int rule_offset = 0;
    int rule_count = 0;
    int first_rule_index = INT_MAX;
};

struct MatchResult
{
    int question_id = 0;
    int image_id = 0;
    int matched_rule_index = -1;
};

struct ModelData
{
    std::vector<RuleRuntime> rules;
    std::vector<std::string> rule_ids;
    std::vector<CueMask> image_masks;
    std::vector<AnswerCueSpan> answer_cue_spans;
    std::vector<AnchorGroup> anchor_groups;
    std::vector<int> candidate_rule_indices;
    std::unordered_map<std::string, uint16_t> answer_to_id;
    std::unordered_map<std::string, uint16_t> cue_to_id;
    std::unordered_map<std::string, uint16_t> keyword_to_id;
    std::unordered_map<std::string, int> keyword_frequency;
    int num_answers = 0;
    int num_cues = 0;
    int max_image_id = 0;
    int max_answer_tokens = 1;
};

struct DeviceBuffers
{
    CompactRule *rules = nullptr;
    CueMask *image_masks = nullptr;
    AnswerCueSpan *answer_cue_spans = nullptr;
    AnchorGroup *anchor_groups = nullptr;
    int *candidate_rule_indices = nullptr;
    EncodedQuestion *questions = nullptr;
    MatchResult *results = nullptr;
    int question_capacity = 0;
};

struct RunConfig
{
    std::string rules_json;
    std::string questions_json;
    std::string annotations_json;
    std::string image_classes_json;
    std::string output_json;
    std::vector<int> gpu_ids;
    int batch_size = kDefaultBatchSize;
};

std::mutex g_log_mutex;

std::string trim_copy(const std::string &input)
{
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin])))
    {
        ++begin;
    }

    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1])))
    {
        --end;
    }
    return input.substr(begin, end - begin);
}

std::string normalize_text(const std::string &input)
{
    std::string out = trim_copy(input);
    for (char &ch : out)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

bool ends_with(const std::string &text, const std::string &suffix)
{
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string singularize_question_token(const std::string &token)
{
    static const std::unordered_map<std::string, std::string> irregular_lemmas = {
        {"people", "person"},
        {"men", "man"},
        {"women", "woman"},
        {"children", "child"},
        {"mice", "mouse"},
        {"geese", "goose"},
        {"teeth", "tooth"},
        {"feet", "foot"},
        {"buses", "bus"},
        {"lenses", "lens"},
    };
    static const std::unordered_set<std::string> no_singularize = {
        "this", "yes", "his", "hers", "ours", "yours", "theirs", "its", "thus",
        "news", "series", "species", "glasses", "sunglasses", "eyeglasses",
        "goggles", "pants", "jeans", "trousers", "shorts", "scissors",
        "binoculars", "pliers", "clothes", "stairs",
    };

    auto irregular_it = irregular_lemmas.find(token);
    if (irregular_it != irregular_lemmas.end())
    {
        return irregular_it->second;
    }
    if (no_singularize.count(token) > 0 || token.size() <= 3)
    {
        return token;
    }
    if (ends_with(token, "ies") && token.size() > 4)
    {
        return token.substr(0, token.size() - 3) + "y";
    }
    if (ends_with(token, "ves") && token.size() > 4)
    {
        std::string stem = token.substr(0, token.size() - 3);
        if (!stem.empty() && stem.back() == 'i')
        {
            stem.back() = 'f';
            return stem + "e";
        }
        return stem + "f";
    }
    if (ends_with(token, "sses") && token.size() > 4)
    {
        return token.substr(0, token.size() - 2);
    }
    if ((ends_with(token, "ches") || ends_with(token, "shes") ||
         ends_with(token, "xes") || ends_with(token, "zes")) &&
        token.size() > 4)
    {
        return token.substr(0, token.size() - 2);
    }
    if (ends_with(token, "es") && token.size() > 4)
    {
        const std::string without_last = token.substr(0, token.size() - 1);
        if (!without_last.empty() && without_last.back() == 'e')
        {
            return without_last;
        }
    }
    if (ends_with(token, "s") &&
        !ends_with(token, "ss") &&
        !ends_with(token, "us") &&
        !ends_with(token, "is") &&
        !ends_with(token, "es"))
    {
        return token.substr(0, token.size() - 1);
    }
    return token;
}

std::string normalize_question_token(const std::string &input)
{
    std::string token = normalize_text(input);
    if (token == "s")
    {
        return "";
    }
    token = singularize_question_token(token);
    return token;
}

std::string json_value_to_string(json_t *value)
{
    if (value == nullptr)
    {
        return "";
    }
    if (json_is_string(value))
    {
        return std::string(json_string_value(value));
    }
    if (json_is_integer(value))
    {
        return std::to_string(static_cast<long long>(json_integer_value(value)));
    }
    if (json_is_real(value))
    {
        std::ostringstream oss;
        oss << json_real_value(value);
        return oss.str();
    }
    return "";
}

double json_value_to_double(json_t *value)
{
    if (value == nullptr)
    {
        return 0.0;
    }
    if (json_is_real(value))
    {
        return json_real_value(value);
    }
    if (json_is_integer(value))
    {
        return static_cast<double>(json_integer_value(value));
    }
    return 0.0;
}

int json_value_to_int(json_t *value, int fallback = 0)
{
    if (value == nullptr)
    {
        return fallback;
    }
    if (json_is_integer(value))
    {
        return static_cast<int>(json_integer_value(value));
    }
    if (json_is_string(value))
    {
        return std::atoi(json_string_value(value));
    }
    return fallback;
}

json_t *expect_array_root(json_t *root, const char *field_name)
{
    if (json_is_object(root))
    {
        json_t *arr = json_object_get(root, field_name);
        if (json_is_array(arr))
        {
            return arr;
        }
        return nullptr;
    }
    if (json_is_array(root))
    {
        return root;
    }
    return nullptr;
}

uint64_t make_group_key(uint16_t answer_id, uint16_t cue_id, uint16_t anchor_keyword_id)
{
    return (static_cast<uint64_t>(answer_id) << 32) |
           (static_cast<uint64_t>(cue_id) << 16) |
           static_cast<uint64_t>(anchor_keyword_id);
}

uint32_t make_answer_cue_key(uint16_t answer_id, uint16_t cue_id)
{
    return (static_cast<uint32_t>(answer_id) << 16) | static_cast<uint32_t>(cue_id);
}

void log_line(const std::string &message)
{
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::cout << message << std::endl;
}

void append_normalized_array_strings(json_t *array, std::vector<std::string> *out)
{
    if (!json_is_array(array))
    {
        return;
    }

    const size_t count = json_array_size(array);
    out->reserve(out->size() + count);
    for (size_t i = 0; i < count; ++i)
    {
        json_t *value = json_array_get(array, i);
        if (!json_is_string(value))
        {
            continue;
        }
        std::string normalized = normalize_text(json_string_value(value));
        if (!normalized.empty())
        {
            out->push_back(std::move(normalized));
        }
    }
}

void append_normalized_keyword_array_strings(json_t *array, std::vector<std::string> *out)
{
    if (!json_is_array(array))
    {
        return;
    }

    const size_t count = json_array_size(array);
    out->reserve(out->size() + count);
    for (size_t i = 0; i < count; ++i)
    {
        json_t *value = json_array_get(array, i);
        if (!json_is_string(value))
        {
            continue;
        }
        std::string normalized = normalize_question_token(json_string_value(value));
        if (!normalized.empty())
        {
            out->push_back(std::move(normalized));
        }
    }
}

void tokenize_to_strings(const std::string &text, std::vector<std::string> *tokens)
{
    tokens->clear();
    std::string current_token;

    auto flush_token = [&]() {
        if (!current_token.empty())
        {
            tokens->push_back(current_token);
            current_token.clear();
        }
    };

    for (char ch : text)
    {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch))
        {
            current_token.push_back(static_cast<char>(std::tolower(uch)));
        }
        else
        {
            flush_token();
            if (ch == '\'')
            {
                tokens->push_back("'");
            }
        }
    }
    flush_token();
}

AnswerCandidateSet build_answer_candidates(const std::string &answer_text, const ModelData &model)
{
    std::vector<uint16_t> candidate_ids;
    const std::string normalized = normalize_text(answer_text);
    auto exact_it = model.answer_to_id.find(normalized);
    if (exact_it != model.answer_to_id.end())
    {
        candidate_ids.push_back(exact_it->second);
    }

    std::vector<std::string> tokens;
    tokenize_to_strings(normalized, &tokens);
    for (size_t start = 0; start < tokens.size(); ++start)
    {
        std::string phrase;
        for (int len = 1; len <= model.max_answer_tokens && start + len <= tokens.size(); ++len)
        {
            if (len == 1)
            {
                phrase = tokens[start];
            }
            else
            {
                phrase.append(" ");
                phrase.append(tokens[start + len - 1]);
            }
            auto it = model.answer_to_id.find(phrase);
            if (it != model.answer_to_id.end())
            {
                candidate_ids.push_back(it->second);
            }
        }
    }

    std::sort(candidate_ids.begin(), candidate_ids.end());
    candidate_ids.erase(std::unique(candidate_ids.begin(), candidate_ids.end()), candidate_ids.end());
    AnswerCandidateSet candidates;
    candidates.count = static_cast<uint16_t>(
        std::min<size_t>(candidate_ids.size(), static_cast<size_t>(kMaxAnswerCandidates)));
    for (size_t i = 0; i < candidates.count; ++i)
    {
        candidates.ids[i] = candidate_ids[i];
    }
    return candidates;
}

template <typename T>
void sort_and_unique(std::vector<T> *items)
{
    std::sort(items->begin(), items->end());
    items->erase(std::unique(items->begin(), items->end()), items->end());
}

void load_rules(const std::string &filename, ModelData *model)
{
    json_error_t error;
    json_t *root = json_load_file(filename.c_str(), 0, &error);
    if (root == nullptr)
    {
        throw std::runtime_error("Failed to load rules JSON: " + std::string(error.text));
    }

    json_t *rules_array = expect_array_root(root, "rules");
    if (rules_array == nullptr)
    {
        json_decref(root);
        throw std::runtime_error("Rules JSON must be an array or {\"rules\": [...]}");
    }

    const size_t count = json_array_size(rules_array);
    std::vector<RawRule> raw_rules;
    raw_rules.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        json_t *rule_obj = json_array_get(rules_array, i);
        if (!json_is_object(rule_obj))
        {
            continue;
        }

        RawRule rule;
        rule.rule_id = normalize_text(json_value_to_string(json_object_get(rule_obj, "rule_id")));
        rule.answer = normalize_text(json_value_to_string(json_object_get(rule_obj, "answer")));
        rule.confidence = json_value_to_double(json_object_get(rule_obj, "confidence"));
        rule.support = json_value_to_double(json_object_get(rule_obj, "support"));
        append_normalized_keyword_array_strings(json_object_get(rule_obj, "text_keywords"), &rule.text_keywords);
        append_normalized_array_strings(json_object_get(rule_obj, "visual_cues"), &rule.visual_cues);
        sort_and_unique(&rule.text_keywords);
        sort_and_unique(&rule.visual_cues);

        if (rule.rule_id.empty())
        {
            rule.rule_id = std::to_string(static_cast<long long>(i + 1));
        }
        raw_rules.push_back(std::move(rule));
    }
    json_decref(root);

    std::sort(raw_rules.begin(), raw_rules.end(), [](const RawRule &lhs, const RawRule &rhs) {
        if (lhs.confidence != rhs.confidence)
        {
            return lhs.confidence > rhs.confidence;
        }
        if (lhs.support != rhs.support)
        {
            return lhs.support > rhs.support;
        }
        return lhs.rule_id < rhs.rule_id;
    });

    for (const RawRule &rule : raw_rules)
    {
        if (!model->answer_to_id.count(rule.answer))
        {
            model->answer_to_id[rule.answer] = static_cast<uint16_t>(model->answer_to_id.size());
        }
        std::vector<std::string> answer_tokens;
        tokenize_to_strings(rule.answer, &answer_tokens);
        model->max_answer_tokens = std::max(model->max_answer_tokens, static_cast<int>(answer_tokens.size()));
        for (const std::string &cue : rule.visual_cues)
        {
            if (!model->cue_to_id.count(cue))
            {
                model->cue_to_id[cue] = static_cast<uint16_t>(model->cue_to_id.size());
            }
        }
        for (const std::string &keyword : rule.text_keywords)
        {
            if (!model->keyword_to_id.count(keyword))
            {
                model->keyword_to_id[keyword] = static_cast<uint16_t>(model->keyword_to_id.size() + 1);
            }
            model->keyword_frequency[keyword] += 1;
        }
    }

    model->num_answers = static_cast<int>(model->answer_to_id.size());
    model->num_cues = static_cast<int>(model->cue_to_id.size());
    if (model->num_cues > kCueMaskWords * 64)
    {
        throw std::runtime_error(
            "Rule cue vocabulary exceeds CUDA matcher capacity: " +
            std::to_string(model->num_cues) + " > " +
            std::to_string(kCueMaskWords * 64));
    }
    model->rules.reserve(raw_rules.size());
    model->rule_ids.reserve(raw_rules.size());

    for (const RawRule &raw_rule : raw_rules)
    {
        RuleRuntime runtime_rule;
        runtime_rule.rule_id = raw_rule.rule_id;
        runtime_rule.compact.answer_id = model->answer_to_id.at(raw_rule.answer);

        std::vector<uint16_t> keyword_ids;
        keyword_ids.reserve(raw_rule.text_keywords.size());
        for (const std::string &keyword : raw_rule.text_keywords)
        {
            keyword_ids.push_back(model->keyword_to_id.at(keyword));
        }
        sort_and_unique(&keyword_ids);

        if (keyword_ids.size() > static_cast<size_t>(kMaxRuleKeywords))
        {
            throw std::runtime_error("Rule has too many keywords after normalization: " + raw_rule.rule_id);
        }

        runtime_rule.compact.num_keywords = static_cast<uint16_t>(keyword_ids.size());
        for (size_t i = 0; i < keyword_ids.size(); ++i)
        {
            runtime_rule.compact.keyword_ids[i] = keyword_ids[i];
        }

        if (!raw_rule.text_keywords.empty())
        {
            std::string anchor_keyword = raw_rule.text_keywords.front();
            int best_frequency = model->keyword_frequency.at(anchor_keyword);
            for (const std::string &keyword : raw_rule.text_keywords)
            {
                int frequency = model->keyword_frequency.at(keyword);
                if (frequency < best_frequency || (frequency == best_frequency && keyword < anchor_keyword))
                {
                    anchor_keyword = keyword;
                    best_frequency = frequency;
                }
            }
            runtime_rule.anchor_keyword_id = model->keyword_to_id.at(anchor_keyword);
        }

        runtime_rule.cue_ids.reserve(raw_rule.visual_cues.size());
        for (const std::string &cue : raw_rule.visual_cues)
        {
            runtime_rule.cue_ids.push_back(model->cue_to_id.at(cue));
        }
        sort_and_unique(&runtime_rule.cue_ids);

        model->rule_ids.push_back(runtime_rule.rule_id);
        model->rules.push_back(std::move(runtime_rule));
    }

    std::unordered_map<uint64_t, std::vector<int>> grouped_rules;
    grouped_rules.reserve(model->rules.size() * 2);
    for (size_t rule_index = 0; rule_index < model->rules.size(); ++rule_index)
    {
        const RuleRuntime &rule = model->rules[rule_index];
        for (uint16_t cue_id : rule.cue_ids)
        {
            grouped_rules[make_group_key(rule.compact.answer_id, cue_id, rule.anchor_keyword_id)]
                .push_back(static_cast<int>(rule_index));
        }
    }

    struct TempGroup
    {
        uint16_t anchor_keyword_id = kNoKeywordId;
        std::vector<int> rule_indices;
    };

    std::unordered_map<uint32_t, std::vector<TempGroup>> answer_cue_groups;
    answer_cue_groups.reserve(grouped_rules.size());
    for (auto &entry : grouped_rules)
    {
        const uint64_t key = entry.first;
        const uint16_t answer_id = static_cast<uint16_t>((key >> 32) & 0xFFFF);
        const uint16_t cue_id = static_cast<uint16_t>((key >> 16) & 0xFFFF);
        const uint16_t anchor_keyword_id = static_cast<uint16_t>(key & 0xFFFF);
        auto &groups = answer_cue_groups[make_answer_cue_key(answer_id, cue_id)];
        TempGroup group;
        group.anchor_keyword_id = anchor_keyword_id;
        group.rule_indices = std::move(entry.second);
        groups.push_back(std::move(group));
    }

    model->answer_cue_spans.assign(static_cast<size_t>(model->num_answers) * model->num_cues, {});
    for (int answer_id = 0; answer_id < model->num_answers; ++answer_id)
    {
        for (int cue_id = 0; cue_id < model->num_cues; ++cue_id)
        {
            auto it = answer_cue_groups.find(make_answer_cue_key(static_cast<uint16_t>(answer_id),
                                                                 static_cast<uint16_t>(cue_id)));
            if (it == answer_cue_groups.end())
            {
                continue;
            }

            auto &groups = it->second;
            std::sort(groups.begin(), groups.end(), [](const TempGroup &lhs, const TempGroup &rhs) {
                return lhs.rule_indices.front() < rhs.rule_indices.front();
            });

            AnswerCueSpan span;
            span.group_offset = static_cast<int>(model->anchor_groups.size());
            span.group_count = static_cast<int>(groups.size());
            model->answer_cue_spans[static_cast<size_t>(answer_id) * model->num_cues + cue_id] = span;

            for (const TempGroup &group : groups)
            {
                AnchorGroup anchor_group;
                anchor_group.anchor_keyword_id = group.anchor_keyword_id;
                anchor_group.rule_offset = static_cast<int>(model->candidate_rule_indices.size());
                anchor_group.rule_count = static_cast<int>(group.rule_indices.size());
                anchor_group.first_rule_index = group.rule_indices.front();
                model->anchor_groups.push_back(anchor_group);
                model->candidate_rule_indices.insert(model->candidate_rule_indices.end(),
                                                     group.rule_indices.begin(),
                                                     group.rule_indices.end());
            }
        }
    }

    log_line("Loaded " + std::to_string(model->rules.size()) + " rules, " +
             std::to_string(model->num_answers) + " answers, " +
             std::to_string(model->num_cues) + " visual cues, " +
             std::to_string(model->keyword_to_id.size()) + " keywords");
}

void set_cue_bit(CueMask *mask, uint16_t cue_id)
{
    const int word_index = cue_id / 64;
    const int bit_index = cue_id % 64;
    if (word_index >= 0 && word_index < kCueMaskWords)
    {
        mask->words[word_index] |= (1ULL << bit_index);
    }
}

void load_image_masks(const std::string &filename, ModelData *model)
{
    json_error_t error;
    json_t *root = json_load_file(filename.c_str(), 0, &error);
    if (root == nullptr)
    {
        throw std::runtime_error("Failed to load detections JSON: " + std::string(error.text));
    }

    auto process_detection_object = [&](json_t *img_obj) {
        if (!json_is_object(img_obj))
        {
            return;
        }

        const int image_id = json_value_to_int(json_object_get(img_obj, "image_id"), -1);
        if (image_id <= 0)
        {
            return;
        }

        if (image_id >= static_cast<int>(model->image_masks.size()))
        {
            model->image_masks.resize(static_cast<size_t>(image_id) + 1);
        }
        model->max_image_id = std::max(model->max_image_id, image_id);

        CueMask &mask = model->image_masks[image_id];
        json_t *classes_array = json_object_get(img_obj, "classes");
        if (!json_is_array(classes_array))
        {
            return;
        }

        const size_t class_count = json_array_size(classes_array);
        for (size_t i = 0; i < class_count; ++i)
        {
            json_t *cls = json_array_get(classes_array, i);
            if (!json_is_string(cls))
            {
                continue;
            }
            std::string class_name = normalize_text(json_string_value(cls));
            auto it = model->cue_to_id.find(class_name);
            if (it != model->cue_to_id.end())
            {
                set_cue_bit(&mask, it->second);
            }
        }
    };

    if (json_is_object(root))
    {
        json_t *detections_array = json_object_get(root, "detections");
        if (json_is_array(detections_array))
        {
            const size_t count = json_array_size(detections_array);
            for (size_t i = 0; i < count; ++i)
            {
                process_detection_object(json_array_get(detections_array, i));
            }
        }
        else
        {
            const char *key = nullptr;
            json_t *value = nullptr;
            json_object_foreach(root, key, value)
            {
                process_detection_object(value);
            }
        }
    }
    else if (json_is_array(root))
    {
        const size_t count = json_array_size(root);
        for (size_t i = 0; i < count; ++i)
        {
            process_detection_object(json_array_get(root, i));
        }
    }
    else
    {
        json_decref(root);
        throw std::runtime_error("Detections JSON must be an array, a mapping, or {\"detections\": [...]}");
    }

    json_decref(root);
    if (model->image_masks.empty())
    {
        model->image_masks.resize(1);
    }

    log_line("Loaded image cue masks for " + std::to_string(model->max_image_id) + " dense image ids");
}

std::vector<AnswerCandidateSet> load_annotation_answer_candidates(
    const std::string &filename,
    const ModelData &model)
{
    json_error_t error;
    json_t *root = json_load_file(filename.c_str(), 0, &error);
    if (root == nullptr)
    {
        throw std::runtime_error("Failed to load annotations JSON: " + std::string(error.text));
    }

    json_t *annotations_array = expect_array_root(root, "annotations");
    if (annotations_array == nullptr)
    {
        json_decref(root);
        throw std::runtime_error("Annotations JSON must be an array or {\"annotations\": [...]}");
    }

    std::vector<AnswerCandidateSet> answer_candidates(1);
    const size_t count = json_array_size(annotations_array);

    for (size_t i = 0; i < count; ++i)
    {
        json_t *ann_obj = json_array_get(annotations_array, i);
        if (!json_is_object(ann_obj))
        {
            continue;
        }

        const int question_id = json_value_to_int(json_object_get(ann_obj, "question_id"), -1);
        if (question_id <= 0)
        {
            continue;
        }
        if (question_id >= static_cast<int>(answer_candidates.size()))
        {
            answer_candidates.resize(static_cast<size_t>(question_id) + 1);
        }

        std::string answer_text;
        json_t *answer = json_object_get(ann_obj, "answer");
        if (json_is_string(answer))
        {
            answer_text = normalize_text(json_string_value(answer));
        }
        if (answer_text.empty())
        {
            json_t *answers_array = json_object_get(ann_obj, "answers");
            if (json_is_array(answers_array) && json_array_size(answers_array) > 0)
            {
                json_t *first_answer = json_array_get(answers_array, 0);
                if (json_is_object(first_answer))
                {
                    answer_text = normalize_text(json_value_to_string(json_object_get(first_answer, "answer")));
                }
                else if (json_is_string(first_answer))
                {
                    answer_text = normalize_text(json_string_value(first_answer));
                }
            }
        }
        if (answer_text.empty())
        {
            answer_text = normalize_text(json_value_to_string(json_object_get(ann_obj, "multiple_choice_answer")));
        }

        answer_candidates[question_id] = build_answer_candidates(answer_text, model);
    }

    json_decref(root);
    return answer_candidates;
}

void tokenize_question(
    const std::string &text,
    const std::unordered_map<std::string, uint16_t> &keyword_to_id,
    EncodedQuestion *question)
{
    std::vector<uint16_t> token_ids;
    token_ids.reserve(16);
    std::string current_token;

    auto flush_token = [&]() {
        if (current_token.empty())
        {
            return;
        }
        const std::string normalized_token = normalize_question_token(current_token);
        auto it = keyword_to_id.find(normalized_token);
        if (it != keyword_to_id.end())
        {
            token_ids.push_back(it->second);
        }
        current_token.clear();
    };

    for (char ch : text)
    {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch))
        {
            current_token.push_back(static_cast<char>(std::tolower(uch)));
        }
        else if (ch == '\'')
        {
            flush_token();
            continue;
        }
        else
        {
            flush_token();
        }
    }
    flush_token();

    sort_and_unique(&token_ids);
    question->num_tokens = static_cast<uint16_t>(
        std::min<size_t>(token_ids.size(), static_cast<size_t>(kMaxQuestionTokens)));
    for (size_t i = 0; i < question->num_tokens; ++i)
    {
        question->token_ids[i] = token_ids[i];
    }
}

std::vector<EncodedQuestion> load_questions(
    const std::string &filename,
    const ModelData &model,
    const std::vector<AnswerCandidateSet> *answer_candidates_by_question)
{
    json_error_t error;
    json_t *root = json_load_file(filename.c_str(), 0, &error);
    if (root == nullptr)
    {
        throw std::runtime_error("Failed to load questions JSON: " + std::string(error.text));
    }

    json_t *questions_array = expect_array_root(root, "questions");
    if (questions_array == nullptr)
    {
        json_decref(root);
        throw std::runtime_error("Questions JSON must be an array or {\"questions\": [...]}");
    }

    const size_t count = json_array_size(questions_array);
    std::vector<EncodedQuestion> questions;
    questions.reserve(count);
    size_t missing_answers = 0;

    for (size_t i = 0; i < count; ++i)
    {
        json_t *question_obj = json_array_get(questions_array, i);
        if (!json_is_object(question_obj))
        {
            continue;
        }

        EncodedQuestion question;
        question.question_id = json_value_to_int(json_object_get(question_obj, "question_id"), 0);
        question.image_id = json_value_to_int(json_object_get(question_obj, "image_id"), 0);

        std::string answer_text = normalize_text(json_value_to_string(json_object_get(question_obj, "answer")));
        if (!answer_text.empty())
        {
            AnswerCandidateSet candidates = build_answer_candidates(answer_text, model);
            question.num_answer_candidates = candidates.count;
            for (size_t j = 0; j < candidates.count; ++j)
            {
                question.answer_ids[j] = candidates.ids[j];
            }
        }
        else if (answer_candidates_by_question != nullptr &&
            question.question_id > 0 &&
            question.question_id < static_cast<int>(answer_candidates_by_question->size()))
        {
            const AnswerCandidateSet &candidates = (*answer_candidates_by_question)[question.question_id];
            question.num_answer_candidates = candidates.count;
            for (size_t j = 0; j < candidates.count; ++j)
            {
                question.answer_ids[j] = candidates.ids[j];
            }
        }
        if (question.num_answer_candidates == 0)
        {
            ++missing_answers;
        }

        std::string question_text = json_value_to_string(json_object_get(question_obj, "question_text"));
        if (question_text.empty())
        {
            question_text = json_value_to_string(json_object_get(question_obj, "question"));
        }
        tokenize_question(question_text, model.keyword_to_id, &question);
        questions.push_back(question);
    }

    json_decref(root);
    log_line("Loaded " + std::to_string(questions.size()) + " questions; " +
             std::to_string(missing_answers) + " questions have answers outside the rule vocabulary");
    return questions;
}

void json_write_escaped_string(FILE *fp, const std::string &value)
{
    for (char ch : value)
    {
        switch (ch)
        {
        case '\\':
            fputs("\\\\", fp);
            break;
        case '"':
            fputs("\\\"", fp);
            break;
        case '\b':
            fputs("\\b", fp);
            break;
        case '\f':
            fputs("\\f", fp);
            break;
        case '\n':
            fputs("\\n", fp);
            break;
        case '\r':
            fputs("\\r", fp);
            break;
        case '\t':
            fputs("\\t", fp);
            break;
        default:
            fputc(ch, fp);
            break;
        }
    }
}

void save_results_to_json(
    const std::string &filename,
    const std::vector<MatchResult> &results,
    const std::vector<std::string> &rule_ids)
{
    FILE *fp = std::fopen(filename.c_str(), "w");
    if (fp == nullptr)
    {
        throw std::runtime_error("Failed to open output file: " + filename);
    }

    fputs("{\"results\":[", fp);
    for (size_t i = 0; i < results.size(); ++i)
    {
        if (i > 0)
        {
            fputc(',', fp);
        }
        fputs("\n{\"question_id\":", fp);
        std::fprintf(fp, "%d", results[i].question_id);
        fputs(",\"image_id\":", fp);
        std::fprintf(fp, "%d", results[i].image_id);
        fputs(",\"rule_id\":\"", fp);
        if (results[i].matched_rule_index >= 0 &&
            results[i].matched_rule_index < static_cast<int>(rule_ids.size()))
        {
            json_write_escaped_string(fp, rule_ids[results[i].matched_rule_index]);
        }
        else
        {
            fputc('0', fp);
        }
        fputs("\"}", fp);
    }
    fputs("\n]}\n", fp);
    std::fclose(fp);
}

__device__ __forceinline__ bool question_has_token(const EncodedQuestion &question, uint16_t token_id)
{
    for (int i = 0; i < question.num_tokens; ++i)
    {
        if (question.token_ids[i] == token_id)
        {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool rule_matches_question(
    const EncodedQuestion &question,
    const CompactRule &rule)
{
    for (int i = 0; i < rule.num_keywords; ++i)
    {
        if (!question_has_token(question, rule.keyword_ids[i]))
        {
            return false;
        }
    }
    return true;
}

__global__ void match_rules_kernel(
    const EncodedQuestion *questions,
    int num_questions,
    const CompactRule *rules,
    const CueMask *image_masks,
    int image_mask_count,
    const AnswerCueSpan *answer_cue_spans,
    const AnchorGroup *anchor_groups,
    const int *candidate_rule_indices,
    int num_cues,
    MatchResult *results)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_questions)
    {
        return;
    }

    const EncodedQuestion question = questions[idx];
    MatchResult result;
    result.question_id = question.question_id;
    result.image_id = question.image_id;
    result.matched_rule_index = -1;

    if (question.num_answer_candidates == 0 ||
        question.image_id <= 0 ||
        question.image_id >= image_mask_count)
    {
        results[idx] = result;
        return;
    }

    const CueMask mask = image_masks[question.image_id];
    int best_rule_index = INT_MAX;

    for (int word_index = 0; word_index < kCueMaskWords; ++word_index)
    {
        uint64_t bits = mask.words[word_index];
        while (bits != 0ULL)
        {
            const int bit = __ffsll(static_cast<long long>(bits)) - 1;
            bits &= (bits - 1);

            const int cue_id = word_index * 64 + bit;
            if (cue_id >= num_cues)
            {
                continue;
            }

            for (int answer_idx = 0; answer_idx < question.num_answer_candidates; ++answer_idx)
            {
                const uint16_t answer_id = question.answer_ids[answer_idx];
                const AnswerCueSpan span = answer_cue_spans[static_cast<int>(answer_id) * num_cues + cue_id];
                for (int group_idx = 0; group_idx < span.group_count; ++group_idx)
                {
                    const AnchorGroup group = anchor_groups[span.group_offset + group_idx];
                    if (group.first_rule_index >= best_rule_index)
                    {
                        continue;
                    }
                    if (group.anchor_keyword_id != kNoKeywordId &&
                        !question_has_token(question, group.anchor_keyword_id))
                    {
                        continue;
                    }

                    const int group_end = group.rule_offset + group.rule_count;
                    for (int rule_ptr = group.rule_offset; rule_ptr < group_end; ++rule_ptr)
                    {
                        const int rule_index = candidate_rule_indices[rule_ptr];
                        if (rule_index >= best_rule_index)
                        {
                            break;
                        }
                        if (rule_matches_question(question, rules[rule_index]))
                        {
                            best_rule_index = rule_index;
                            break;
                        }
                    }
                }
            }
        }
    }

    if (best_rule_index != INT_MAX)
    {
        result.matched_rule_index = best_rule_index;
    }
    results[idx] = result;
}

void ensure_question_capacity(DeviceBuffers *buffers, int capacity)
{
    if (capacity <= buffers->question_capacity)
    {
        return;
    }
    if (buffers->questions != nullptr)
    {
        CUDA_CHECK(cudaFree(buffers->questions));
    }
    if (buffers->results != nullptr)
    {
        CUDA_CHECK(cudaFree(buffers->results));
    }
    CUDA_CHECK(cudaMalloc(&buffers->questions, static_cast<size_t>(capacity) * sizeof(EncodedQuestion)));
    CUDA_CHECK(cudaMalloc(&buffers->results, static_cast<size_t>(capacity) * sizeof(MatchResult)));
    buffers->question_capacity = capacity;
}

void allocate_static_buffers(int device_id, const ModelData &model, DeviceBuffers *buffers)
{
    CUDA_CHECK(cudaSetDevice(device_id));
    std::vector<CompactRule> compact_rules;
    compact_rules.reserve(model.rules.size());
    for (const RuleRuntime &rule : model.rules)
    {
        compact_rules.push_back(rule.compact);
    }

    CUDA_CHECK(cudaMalloc(&buffers->rules, compact_rules.size() * sizeof(CompactRule)));
    CUDA_CHECK(cudaMemcpy(buffers->rules,
                          compact_rules.data(),
                          compact_rules.size() * sizeof(CompactRule),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&buffers->image_masks, model.image_masks.size() * sizeof(CueMask)));
    CUDA_CHECK(cudaMemcpy(buffers->image_masks,
                          model.image_masks.data(),
                          model.image_masks.size() * sizeof(CueMask),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&buffers->answer_cue_spans,
                          model.answer_cue_spans.size() * sizeof(AnswerCueSpan)));
    CUDA_CHECK(cudaMemcpy(buffers->answer_cue_spans,
                          model.answer_cue_spans.data(),
                          model.answer_cue_spans.size() * sizeof(AnswerCueSpan),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&buffers->anchor_groups, model.anchor_groups.size() * sizeof(AnchorGroup)));
    CUDA_CHECK(cudaMemcpy(buffers->anchor_groups,
                          model.anchor_groups.data(),
                          model.anchor_groups.size() * sizeof(AnchorGroup),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&buffers->candidate_rule_indices,
                          model.candidate_rule_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(buffers->candidate_rule_indices,
                          model.candidate_rule_indices.data(),
                          model.candidate_rule_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
}

void free_device_buffers(int device_id, DeviceBuffers *buffers)
{
    CUDA_CHECK(cudaSetDevice(device_id));
    if (buffers->rules != nullptr)
    {
        CUDA_CHECK(cudaFree(buffers->rules));
    }
    if (buffers->image_masks != nullptr)
    {
        CUDA_CHECK(cudaFree(buffers->image_masks));
    }
    if (buffers->answer_cue_spans != nullptr)
    {
        CUDA_CHECK(cudaFree(buffers->answer_cue_spans));
    }
    if (buffers->anchor_groups != nullptr)
    {
        CUDA_CHECK(cudaFree(buffers->anchor_groups));
    }
    if (buffers->candidate_rule_indices != nullptr)
    {
        CUDA_CHECK(cudaFree(buffers->candidate_rule_indices));
    }
    if (buffers->questions != nullptr)
    {
        CUDA_CHECK(cudaFree(buffers->questions));
    }
    if (buffers->results != nullptr)
    {
        CUDA_CHECK(cudaFree(buffers->results));
    }
}

void run_device_worker(
    int device_id,
    const ModelData &model,
    const std::vector<EncodedQuestion> &questions,
    int start_index,
    int end_index,
    int batch_size,
    std::vector<MatchResult> *results,
    std::atomic<int64_t> *processed_questions)
{
    CUDA_CHECK(cudaSetDevice(device_id));
    DeviceBuffers buffers;
    allocate_static_buffers(device_id, model, &buffers);

    std::vector<MatchResult> host_batch_results;
    host_batch_results.reserve(batch_size);
    const int total = end_index - start_index;
    const int total_batches = (total + batch_size - 1) / batch_size;

    for (int batch_start = start_index, batch_id = 0; batch_start < end_index; batch_start += batch_size, ++batch_id)
    {
        const int current_batch = std::min(batch_size, end_index - batch_start);
        ensure_question_capacity(&buffers, current_batch);

        CUDA_CHECK(cudaMemcpy(buffers.questions,
                              questions.data() + batch_start,
                              static_cast<size_t>(current_batch) * sizeof(EncodedQuestion),
                              cudaMemcpyHostToDevice));

        const int blocks = (current_batch + kThreadsPerBlock - 1) / kThreadsPerBlock;
        match_rules_kernel<<<blocks, kThreadsPerBlock>>>(
            buffers.questions,
            current_batch,
            buffers.rules,
            buffers.image_masks,
            static_cast<int>(model.image_masks.size()),
            buffers.answer_cue_spans,
            buffers.anchor_groups,
            buffers.candidate_rule_indices,
            model.num_cues,
            buffers.results);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        host_batch_results.resize(current_batch);
        CUDA_CHECK(cudaMemcpy(host_batch_results.data(),
                              buffers.results,
                              static_cast<size_t>(current_batch) * sizeof(MatchResult),
                              cudaMemcpyDeviceToHost));
        std::copy(host_batch_results.begin(),
                  host_batch_results.end(),
                  results->begin() + batch_start);

        const int64_t done = processed_questions->fetch_add(current_batch) + current_batch;
        log_line("GPU " + std::to_string(device_id) + " finished batch " +
                 std::to_string(batch_id + 1) + "/" + std::to_string(total_batches) +
                 " (" + std::to_string(done) + " questions done globally)");
    }

    free_device_buffers(device_id, &buffers);
}

std::vector<int> parse_gpu_ids(const std::string &csv, int detected_gpu_count)
{
    std::vector<int> gpu_ids;
    if (csv.empty())
    {
        for (int i = 0; i < detected_gpu_count; ++i)
        {
            gpu_ids.push_back(i);
        }
        return gpu_ids;
    }

    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (token.empty())
        {
            continue;
        }
        int gpu_id = std::stoi(token);
        if (gpu_id < 0 || gpu_id >= detected_gpu_count)
        {
            throw std::runtime_error("GPU id out of range: " + token);
        }
        gpu_ids.push_back(gpu_id);
    }

    if (gpu_ids.empty())
    {
        throw std::runtime_error("No usable GPUs were selected");
    }
    return gpu_ids;
}

std::string infer_default_annotations_path(const std::string &questions_path)
{
    const size_t slash = questions_path.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? "." : questions_path.substr(0, slash);
    return dir + "/annotations.json";
}

RunConfig parse_args(int argc, char **argv)
{
    RunConfig config;
    std::string gpu_csv;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto read_value = [&](const char *flag_name) -> std::string {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(std::string("Missing value for ") + flag_name);
            }
            return argv[++i];
        };

        if (arg == "--rules_json")
        {
            config.rules_json = read_value("--rules_json");
        }
        else if (arg == "--questions_json")
        {
            config.questions_json = read_value("--questions_json");
        }
        else if (arg == "--annotations_json")
        {
            config.annotations_json = read_value("--annotations_json");
        }
        else if (arg == "--image_classes_json")
        {
            config.image_classes_json = read_value("--image_classes_json");
        }
        else if (arg == "--output_json")
        {
            config.output_json = read_value("--output_json");
        }
        else if (arg == "--gpus")
        {
            gpu_csv = read_value("--gpus");
        }
        else if (arg == "--batch_size")
        {
            config.batch_size = std::stoi(read_value("--batch_size"));
        }
        else if (arg == "--help")
        {
            std::cout
                << "Usage: cuda_matcher --rules_json RULES --questions_json QUESTIONS "
                << "--image_classes_json DETS --output_json OUT "
                << "[--annotations_json ANNS] [--gpus 0,1,2,3] [--batch_size 262144]\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.rules_json.empty() || config.questions_json.empty() ||
        config.image_classes_json.empty() || config.output_json.empty())
    {
        throw std::runtime_error("Missing required arguments. Use --help for usage.");
    }
    if (config.batch_size <= 0)
    {
        throw std::runtime_error("batch_size must be positive");
    }

    int detected_gpu_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&detected_gpu_count));
    if (detected_gpu_count <= 0)
    {
        throw std::runtime_error("No CUDA devices detected");
    }
    config.gpu_ids = parse_gpu_ids(gpu_csv, detected_gpu_count);
    return config;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const RunConfig config = parse_args(argc, argv);
        log_line("Using GPUs: " + [&]() {
            std::ostringstream oss;
            for (size_t i = 0; i < config.gpu_ids.size(); ++i)
            {
                if (i > 0)
                {
                    oss << ",";
                }
                oss << config.gpu_ids[i];
            }
            return oss.str();
        }());

        ModelData model;
        load_rules(config.rules_json, &model);
        load_image_masks(config.image_classes_json, &model);

        std::vector<AnswerCandidateSet> answer_candidates_by_question;
        if (!config.annotations_json.empty())
        {
            answer_candidates_by_question = load_annotation_answer_candidates(config.annotations_json, model);
        }
        else
        {
            const std::string inferred_annotations = infer_default_annotations_path(config.questions_json);
            FILE *fp = std::fopen(inferred_annotations.c_str(), "r");
            if (fp != nullptr)
            {
                std::fclose(fp);
                log_line("No annotations path supplied; using sibling file " + inferred_annotations);
                answer_candidates_by_question = load_annotation_answer_candidates(inferred_annotations, model);
            }
        }

        std::vector<EncodedQuestion> questions = load_questions(
            config.questions_json,
            model,
            answer_candidates_by_question.empty() ? nullptr : &answer_candidates_by_question);
        std::vector<MatchResult> results(questions.size());

        std::atomic<int64_t> processed_questions(0);
        std::vector<std::thread> workers;
        workers.reserve(config.gpu_ids.size());

        const int total_questions = static_cast<int>(questions.size());
        const int num_workers = static_cast<int>(config.gpu_ids.size());
        for (int worker_idx = 0; worker_idx < num_workers; ++worker_idx)
        {
            const int start_index = static_cast<int>(
                (static_cast<int64_t>(total_questions) * worker_idx) / num_workers);
            const int end_index = static_cast<int>(
                (static_cast<int64_t>(total_questions) * (worker_idx + 1)) / num_workers);
            workers.emplace_back(run_device_worker,
                                 config.gpu_ids[worker_idx],
                                 std::cref(model),
                                 std::cref(questions),
                                 start_index,
                                 end_index,
                                 config.batch_size,
                                 &results,
                                 &processed_questions);
        }

        for (std::thread &worker : workers)
        {
            worker.join();
        }

        int64_t matched_questions = 0;
        for (const MatchResult &result : results)
        {
            if (result.matched_rule_index >= 0)
            {
                ++matched_questions;
            }
        }

        save_results_to_json(config.output_json, results, model.rule_ids);
        log_line("Matched " + std::to_string(matched_questions) + " / " +
                 std::to_string(results.size()) + " questions (" +
                 std::to_string(static_cast<double>(matched_questions) /
                                std::max<size_t>(results.size(), 1)) +
                 ")");
        log_line("Matching completed for " + std::to_string(results.size()) +
                 " questions; results saved to " + config.output_json);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
