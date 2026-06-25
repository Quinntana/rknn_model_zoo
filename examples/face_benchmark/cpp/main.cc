// Dedicated RKNN face-model benchmark.
//
// This tool intentionally avoids RKNN_FLAG_COLLECT_PERF_MASK. Its headline
// speed metric is low-overhead wall time for rknn_run + rknn_outputs_get on a
// selected single NPU core.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

#include "image_utils.h"
#include "rknn_api.h"
#include "rknn_box_priors.h"

struct CliOptions {
    std::string models_path = "model/model_registry.yaml";
    std::string image_path = "model/test.jpg";
    bool run_speed = false;
    bool run_accuracy = false;
    std::string task = "all";
    int core = 0;
    int warmup = 20;
    int iters = 200;
    std::string dataset = "internal";
    std::string manifest_path;
    std::string output_path;
};

struct ModelEntry {
    std::string classification;
    std::string task;
    std::string demo;
    std::string name;
    std::string path;
    std::string resolved_path;
    std::vector<int> input_shape;
    std::string input_layout = "nchw";
    std::string input_type = "uint8";
    std::string dtype = "INT8";
    std::string color_order = "rgb";
    std::string resize = "letterbox";
    std::string output_decoder = "unsupported";
    std::string output_activation = "none";
    std::string feature_norm = "l2";
    int positive_index = 1;
    int output_index = 0;
    double threshold = 0.5;
    std::vector<double> mean;
    std::vector<double> std;
};

struct TimingSample {
    double run_ms = 0.0;
    double outputs_get_ms = 0.0;
    double model_ms = 0.0;
};

struct TimingSummary {
    double avg = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
};

struct OutputBlob {
    std::vector<float> values;
    std::vector<int> dims;
};

struct InferenceResult {
    std::vector<OutputBlob> outputs;
    TimingSample timing;
};

struct Detection {
    std::string image;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
};

struct Box {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

static Box make_box(float x1, float y1, float x2, float y2)
{
    Box box;
    box.x1 = x1;
    box.y1 = y1;
    box.x2 = x2;
    box.y2 = y2;
    return box;
}

struct SpeedRow {
    ModelEntry model;
    TimingSummary ms;
};

struct AccuracyRow {
    ModelEntry model;
    std::string dataset;
    std::string metric_summary;
    std::vector<std::pair<std::string, std::string> > csv_metrics;
};

static double now_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static std::string trim(const std::string &s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static bool starts_with(const std::string &s, const std::string &prefix)
{
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static bool file_exists(const std::string &path)
{
    std::ifstream f(path.c_str(), std::ios::binary);
    return f.good();
}

static std::string dirname_of(const std::string &path)
{
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return ".";
    }
    if (slash == 0) {
        return "/";
    }
    return path.substr(0, slash);
}

static bool is_absolute_path(const std::string &path)
{
    return !path.empty() && path[0] == '/';
}

static std::string join_path(const std::string &a, const std::string &b)
{
    if (a.empty() || a == ".") {
        return b;
    }
    if (a[a.size() - 1] == '/') {
        return a + b;
    }
    return a + "/" + b;
}

static std::string strip_quotes(std::string value)
{
    value = trim(value);
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

static std::vector<std::string> split_ws(const std::string &line)
{
    std::istringstream ss(line);
    std::vector<std::string> out;
    std::string token;
    while (ss >> token) {
        out.push_back(token);
    }
    return out;
}

static std::vector<std::string> split_char(const std::string &value, char sep)
{
    std::vector<std::string> out;
    std::string item;
    std::istringstream ss(value);
    while (std::getline(ss, item, sep)) {
        out.push_back(trim(item));
    }
    return out;
}

static std::vector<int> parse_int_list(std::string value)
{
    value = trim(value);
    if (!value.empty() && value.front() == '[') {
        value.erase(value.begin());
    }
    if (!value.empty() && value.back() == ']') {
        value.pop_back();
    }
    std::vector<std::string> parts = split_char(value, ',');
    std::vector<int> out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (!parts[i].empty()) {
            out.push_back(std::atoi(parts[i].c_str()));
        }
    }
    return out;
}

static std::vector<double> parse_double_list(std::string value)
{
    value = trim(value);
    if (!value.empty() && value.front() == '[') {
        value.erase(value.begin());
    }
    if (!value.empty() && value.back() == ']') {
        value.pop_back();
    }
    std::vector<std::string> parts = split_char(value, ',');
    std::vector<double> out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (!parts[i].empty()) {
            out.push_back(std::atof(parts[i].c_str()));
        }
    }
    return out;
}

static void assign_model_key(ModelEntry *model, const std::string &key, const std::string &raw_value)
{
    std::string value = strip_quotes(raw_value);
    if (key == "classification") {
        model->classification = value;
    } else if (key == "task") {
        model->task = value;
    } else if (key == "demo") {
        model->demo = value;
    } else if (key == "name") {
        model->name = value;
    } else if (key == "path") {
        model->path = value;
    } else if (key == "input_shape") {
        model->input_shape = parse_int_list(value);
    } else if (key == "input_layout") {
        model->input_layout = lower(value);
    } else if (key == "input_type") {
        model->input_type = lower(value);
    } else if (key == "dtype") {
        model->dtype = value;
    } else if (key == "color_order") {
        model->color_order = lower(value);
    } else if (key == "resize") {
        model->resize = lower(value);
    } else if (key == "output_decoder") {
        model->output_decoder = lower(value);
    } else if (key == "output_activation") {
        model->output_activation = lower(value);
    } else if (key == "feature_norm") {
        model->feature_norm = lower(value);
    } else if (key == "positive_index") {
        model->positive_index = std::atoi(value.c_str());
    } else if (key == "output_index") {
        model->output_index = std::atoi(value.c_str());
    } else if (key == "threshold") {
        model->threshold = std::atof(value.c_str());
    } else if (key == "mean") {
        model->mean = parse_double_list(value);
    } else if (key == "std") {
        model->std = parse_double_list(value);
    }
}

static bool parse_key_value(const std::string &line, std::string *key, std::string *value)
{
    size_t colon = line.find(':');
    if (colon == std::string::npos) {
        return false;
    }
    *key = trim(line.substr(0, colon));
    *value = trim(line.substr(colon + 1));
    return !key->empty();
}

static bool parse_model_registry(const std::string &path, std::vector<ModelEntry> *models)
{
    std::ifstream in(path.c_str());
    if (!in) {
        printf("[error] failed to open model registry: %s\n", path.c_str());
        return false;
    }

    std::string registry_dir = dirname_of(path);
    std::string line;
    ModelEntry current;
    bool in_model = false;
    while (std::getline(in, line)) {
        size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trim(line);
        if (line.empty() || line == "models:") {
            continue;
        }
        if (starts_with(line, "-")) {
            if (in_model && !current.name.empty()) {
                models->push_back(current);
            }
            current = ModelEntry();
            in_model = true;
            line = trim(line.substr(1));
            if (line.empty()) {
                continue;
            }
        }

        if (!in_model) {
            continue;
        }
        std::string key;
        std::string value;
        if (parse_key_value(line, &key, &value)) {
            assign_model_key(&current, key, value);
        }
    }
    if (in_model && !current.name.empty()) {
        models->push_back(current);
    }

    for (size_t i = 0; i < models->size(); ++i) {
        ModelEntry &m = (*models)[i];
        if (m.classification.empty()) {
            m.classification = m.task;
        }
        if (m.demo.empty()) {
            m.demo = m.task;
        }
        if (m.input_shape.empty()) {
            m.input_shape.push_back(1);
            m.input_shape.push_back(3);
            m.input_shape.push_back(0);
            m.input_shape.push_back(0);
        }
        if (is_absolute_path(m.path)) {
            m.resolved_path = m.path;
        } else if (file_exists(m.path)) {
            m.resolved_path = m.path;
        } else {
            m.resolved_path = join_path(registry_dir, m.path);
        }
    }
    return true;
}

static void print_usage(const char *program)
{
    printf("Usage: %s [--models model/model_registry.yaml] [--speed] [--accuracy]\n", program);
    printf("          [--task face_detection|liveness|feature_extraction|all]\n");
    printf("          [--core 0|1|2] [--warmup 20] [--iters 200]\n");
    printf("          [--image model/test.jpg] [--dataset wider|celeba_spoof|lfw|internal]\n");
    printf("          [--manifest path/to/manifest.txt] [--output results.csv]\n");
}

static bool parse_cli(int argc, char **argv, CliOptions *opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--models" && i + 1 < argc) {
            opt->models_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            opt->image_path = argv[++i];
        } else if (arg == "--speed") {
            opt->run_speed = true;
        } else if (arg == "--accuracy") {
            opt->run_accuracy = true;
        } else if (arg == "--task" && i + 1 < argc) {
            opt->task = argv[++i];
        } else if (arg == "--core" && i + 1 < argc) {
            opt->core = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            opt->warmup = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            opt->iters = std::atoi(argv[++i]);
        } else if (arg == "--dataset" && i + 1 < argc) {
            opt->dataset = argv[++i];
        } else if (arg == "--manifest" && i + 1 < argc) {
            opt->manifest_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            opt->output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            printf("[error] unknown or incomplete argument: %s\n", arg.c_str());
            return false;
        }
    }
    if (!opt->run_speed && !opt->run_accuracy) {
        opt->run_speed = true;
    }
    if (opt->core < 0 || opt->core > 2) {
        printf("[error] --core must be 0, 1, or 2\n");
        return false;
    }
    if (opt->warmup < 0 || opt->iters <= 0) {
        printf("[error] --warmup must be >= 0 and --iters must be > 0\n");
        return false;
    }
    if (opt->task != "all" && opt->task != "face_detection" &&
        opt->task != "liveness" && opt->task != "feature_extraction") {
        printf("[error] unsupported --task: %s\n", opt->task.c_str());
        return false;
    }
    if (opt->run_accuracy && opt->manifest_path.empty()) {
        printf("[error] --accuracy requires --manifest\n");
        return false;
    }
    return true;
}

static rknn_core_mask core_mask_from_index(int core)
{
    if (core == 1) {
        return RKNN_NPU_CORE_1;
    }
    if (core == 2) {
        return RKNN_NPU_CORE_2;
    }
    return RKNN_NPU_CORE_0;
}

static std::string tensor_shape_string(const std::vector<int> &shape)
{
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

static size_t tensor_element_count(const rknn_tensor_attr &attr)
{
    if (attr.n_elems > 0) {
        return attr.n_elems;
    }
    size_t n = 1;
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        n *= static_cast<size_t>(std::max<uint32_t>(1, attr.dims[i]));
    }
    return n;
}

static int model_width_from_entry(const ModelEntry &model, const rknn_tensor_attr &attr)
{
    if (model.input_shape.size() >= 4) {
        return model.input_layout == "nhwc" ? model.input_shape[2] : model.input_shape[3];
    }
    if (attr.n_dims >= 4) {
        return attr.fmt == RKNN_TENSOR_NHWC ? attr.dims[2] : attr.dims[3];
    }
    return 0;
}

static int model_height_from_entry(const ModelEntry &model, const rknn_tensor_attr &attr)
{
    if (model.input_shape.size() >= 4) {
        return model.input_layout == "nhwc" ? model.input_shape[1] : model.input_shape[2];
    }
    if (attr.n_dims >= 4) {
        return attr.fmt == RKNN_TENSOR_NHWC ? attr.dims[1] : attr.dims[2];
    }
    return 0;
}

class ImageHolder {
public:
    ImageHolder() { std::memset(&image_, 0, sizeof(image_)); }
    ~ImageHolder() { reset(); }

    bool load(const std::string &path)
    {
        reset();
        int ret = read_image(path.c_str(), &image_);
        if (ret != 0) {
            printf("[error] failed to read image: %s\n", path.c_str());
            return false;
        }
        return true;
    }

    image_buffer_t *get() { return &image_; }

private:
    void reset()
    {
        if (image_.virt_addr != NULL) {
            free(image_.virt_addr);
        }
        std::memset(&image_, 0, sizeof(image_));
    }

    image_buffer_t image_;
};

class RknnModel {
public:
    RknnModel() {}
    ~RknnModel() { release(); }

    bool init(const ModelEntry &entry, int core)
    {
        entry_ = entry;
        std::ifstream file(entry.resolved_path.c_str(), std::ios::binary);
        if (!file) {
            printf("[error] failed to open RKNN model: %s\n", entry.resolved_path.c_str());
            return false;
        }
        model_data_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (model_data_.empty()) {
            printf("[error] empty RKNN model: %s\n", entry.resolved_path.c_str());
            return false;
        }

        uint32_t init_flags = RKNN_FLAG_ENABLE_SRAM;
        int ret = rknn_init(&ctx_, model_data_.data(), static_cast<uint32_t>(model_data_.size()),
                            init_flags, NULL);
        if (ret != RKNN_SUCC) {
            printf("[error] rknn_init failed for %s ret=%d\n", entry.name.c_str(), ret);
            return false;
        }

        ret = rknn_set_core_mask(ctx_, core_mask_from_index(core));
        if (ret != RKNN_SUCC) {
            printf("[warn] rknn_set_core_mask failed for %s core=%d ret=%d\n",
                   entry.name.c_str(), core, ret);
        }

        ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
        if (ret != RKNN_SUCC) {
            printf("[error] RKNN_QUERY_IN_OUT_NUM failed for %s ret=%d\n", entry.name.c_str(), ret);
            return false;
        }
        input_attrs_.resize(io_num_.n_input);
        output_attrs_.resize(io_num_.n_output);
        for (uint32_t i = 0; i < io_num_.n_input; ++i) {
            std::memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
            input_attrs_[i].index = i;
            ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC) {
                printf("[error] RKNN_QUERY_INPUT_ATTR failed for %s ret=%d\n", entry.name.c_str(), ret);
                return false;
            }
        }
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            std::memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
            output_attrs_[i].index = i;
            ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC) {
                printf("[error] RKNN_QUERY_OUTPUT_ATTR failed for %s ret=%d\n", entry.name.c_str(), ret);
                return false;
            }
        }

        input_width_ = model_width_from_entry(entry, input_attrs_[0]);
        input_height_ = model_height_from_entry(entry, input_attrs_[0]);
        if (input_width_ <= 0 || input_height_ <= 0) {
            printf("[error] invalid input size for %s\n", entry.name.c_str());
            return false;
        }
        return true;
    }

    void release()
    {
        if (ctx_ != 0) {
            rknn_destroy(ctx_);
            ctx_ = 0;
        }
        model_data_.clear();
        input_attrs_.clear();
        output_attrs_.clear();
    }

    bool set_input(image_buffer_t *src, letterbox_t *letterbox)
    {
        uint8_input_.clear();
        float_input_.clear();
        image_buffer_t dst;
        std::memset(&dst, 0, sizeof(dst));
        dst.width = input_width_;
        dst.height = input_height_;
        dst.width_stride = input_width_;
        dst.height_stride = input_height_;
        dst.format = IMAGE_FORMAT_RGB888;
        dst.size = input_width_ * input_height_ * 3;
        uint8_input_.resize(dst.size);
        dst.virt_addr = uint8_input_.data();

        int ret = 0;
        if (entry_.resize == "stretch") {
            ret = convert_image(src, &dst, NULL, NULL, 114);
            if (letterbox != NULL) {
                letterbox->x_pad = 0;
                letterbox->y_pad = 0;
                letterbox->scale = static_cast<float>(input_width_) / std::max(1, src->width);
            }
        } else {
            ret = convert_image_with_letterbox(src, &dst, letterbox, 114);
        }
        if (ret != 0) {
            printf("[error] image resize failed for %s ret=%d\n", entry_.name.c_str(), ret);
            return false;
        }
        if (entry_.color_order == "bgr") {
            for (size_t i = 0; i + 2 < uint8_input_.size(); i += 3) {
                std::swap(uint8_input_[i], uint8_input_[i + 2]);
            }
        }

        rknn_input input;
        std::memset(&input, 0, sizeof(input));
        input.index = 0;
        input.pass_through = 0;
        if (entry_.input_type == "float32" || entry_.input_type == "fp32") {
            prepare_float_input();
            input.buf = float_input_.data();
            input.size = static_cast<uint32_t>(float_input_.size() * sizeof(float));
            input.type = RKNN_TENSOR_FLOAT32;
            input.fmt = (entry_.input_layout == "nchw") ? RKNN_TENSOR_NCHW : RKNN_TENSOR_NHWC;
        } else {
            input.buf = uint8_input_.data();
            input.size = static_cast<uint32_t>(uint8_input_.size());
            input.type = RKNN_TENSOR_UINT8;
            input.fmt = RKNN_TENSOR_NHWC;
        }

        ret = rknn_inputs_set(ctx_, 1, &input);
        if (ret != RKNN_SUCC) {
            printf("[error] rknn_inputs_set failed for %s ret=%d\n", entry_.name.c_str(), ret);
            return false;
        }
        return true;
    }

    bool run_once(bool want_float, InferenceResult *result)
    {
        std::vector<rknn_output> outputs(io_num_.n_output);
        std::memset(outputs.data(), 0, sizeof(rknn_output) * outputs.size());
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            outputs[i].index = i;
            outputs[i].want_float = want_float ? 1 : 0;
        }

        double run_start = now_ms();
        int ret = rknn_run(ctx_, NULL);
        double run_end = now_ms();
        if (ret != RKNN_SUCC) {
            printf("[error] rknn_run failed for %s ret=%d\n", entry_.name.c_str(), ret);
            return false;
        }

        double get_start = now_ms();
        ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), NULL);
        double get_end = now_ms();
        if (ret != RKNN_SUCC) {
            printf("[error] rknn_outputs_get failed for %s ret=%d\n", entry_.name.c_str(), ret);
            return false;
        }

        if (result != NULL) {
            result->timing.run_ms = run_end - run_start;
            result->timing.outputs_get_ms = get_end - get_start;
            result->timing.model_ms = result->timing.run_ms + result->timing.outputs_get_ms;
            result->outputs.clear();
            result->outputs.resize(io_num_.n_output);
            for (uint32_t i = 0; i < io_num_.n_output; ++i) {
                size_t count = tensor_element_count(output_attrs_[i]);
                result->outputs[i].dims.clear();
                for (uint32_t d = 0; d < output_attrs_[i].n_dims; ++d) {
                    result->outputs[i].dims.push_back(output_attrs_[i].dims[d]);
                }
                if (want_float) {
                    float *ptr = static_cast<float *>(outputs[i].buf);
                    result->outputs[i].values.assign(ptr, ptr + count);
                }
            }
        }
        rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
        return true;
    }

    int input_width() const { return input_width_; }
    int input_height() const { return input_height_; }
    const std::vector<rknn_tensor_attr> &output_attrs() const { return output_attrs_; }

private:
    void prepare_float_input()
    {
        const double mean0 = entry_.mean.size() > 0 ? entry_.mean[0] : 0.0;
        const double mean1 = entry_.mean.size() > 1 ? entry_.mean[1] : mean0;
        const double mean2 = entry_.mean.size() > 2 ? entry_.mean[2] : mean0;
        const double std0 = entry_.std.size() > 0 && entry_.std[0] != 0.0 ? entry_.std[0] : 1.0;
        const double std1 = entry_.std.size() > 1 && entry_.std[1] != 0.0 ? entry_.std[1] : std0;
        const double std2 = entry_.std.size() > 2 && entry_.std[2] != 0.0 ? entry_.std[2] : std0;
        float_input_.assign(input_width_ * input_height_ * 3, 0.0f);
        if (entry_.input_layout == "nchw") {
            const size_t plane = static_cast<size_t>(input_width_ * input_height_);
            for (int y = 0; y < input_height_; ++y) {
                for (int x = 0; x < input_width_; ++x) {
                    size_t src = static_cast<size_t>((y * input_width_ + x) * 3);
                    size_t dst = static_cast<size_t>(y * input_width_ + x);
                    float_input_[dst] = static_cast<float>((uint8_input_[src] - mean0) / std0);
                    float_input_[plane + dst] = static_cast<float>((uint8_input_[src + 1] - mean1) / std1);
                    float_input_[2 * plane + dst] = static_cast<float>((uint8_input_[src + 2] - mean2) / std2);
                }
            }
        } else {
            for (size_t i = 0; i + 2 < uint8_input_.size(); i += 3) {
                float_input_[i] = static_cast<float>((uint8_input_[i] - mean0) / std0);
                float_input_[i + 1] = static_cast<float>((uint8_input_[i + 1] - mean1) / std1);
                float_input_[i + 2] = static_cast<float>((uint8_input_[i + 2] - mean2) / std2);
            }
        }
    }

    ModelEntry entry_;
    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_;
    std::vector<char> model_data_;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<unsigned char> uint8_input_;
    std::vector<float> float_input_;
    int input_width_ = 0;
    int input_height_ = 0;
};

static TimingSummary summarize(const std::vector<double> &values)
{
    TimingSummary s;
    if (values.empty()) {
        return s;
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    double total = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    s.avg = total / sorted.size();
    size_t p50_index = static_cast<size_t>((sorted.size() - 1) * 0.50 + 0.5);
    size_t p95_index = static_cast<size_t>((sorted.size() - 1) * 0.95 + 0.5);
    s.p50 = sorted[std::min(p50_index, sorted.size() - 1)];
    s.p95 = sorted[std::min(p95_index, sorted.size() - 1)];
    return s;
}

static bool run_speed_benchmark(const ModelEntry &entry,
                                const CliOptions &opt,
                                SpeedRow *row)
{
    ImageHolder image;
    if (!image.load(opt.image_path)) {
        return false;
    }
    RknnModel model;
    if (!model.init(entry, opt.core)) {
        return false;
    }
    letterbox_t letterbox;
    std::memset(&letterbox, 0, sizeof(letterbox));
    if (!model.set_input(image.get(), &letterbox)) {
        return false;
    }
    for (int i = 0; i < opt.warmup; ++i) {
        if (!model.run_once(false, NULL)) {
            return false;
        }
    }
    std::vector<double> model_ms;
    model_ms.reserve(opt.iters);
    for (int i = 0; i < opt.iters; ++i) {
        InferenceResult result;
        if (!model.run_once(false, &result)) {
            return false;
        }
        model_ms.push_back(result.timing.model_ms);
    }
    row->model = entry;
    row->ms = summarize(model_ms);
    return true;
}

static float overlap_iou(const Box &a, const Box &b)
{
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1 + 1.0f);
    float h = std::max(0.0f, yy2 - yy1 + 1.0f);
    float inter = w * h;
    float area_a = std::max(0.0f, a.x2 - a.x1 + 1.0f) * std::max(0.0f, a.y2 - a.y1 + 1.0f);
    float area_b = std::max(0.0f, b.x2 - b.x1 + 1.0f) * std::max(0.0f, b.y2 - b.y1 + 1.0f);
    float denom = area_a + area_b - inter;
    return denom <= 0.0f ? 0.0f : inter / denom;
}

static std::vector<int> nms_indices(const std::vector<Detection> &dets, float threshold)
{
    std::vector<int> order(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return dets[a].score > dets[b].score;
    });
    std::vector<int> keep;
    std::vector<char> suppressed(dets.size(), 0);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        int i = order[oi];
        if (suppressed[i]) {
            continue;
        }
        keep.push_back(i);
        Box a = make_box(dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            int j = order[oj];
            if (suppressed[j]) {
                continue;
            }
            Box b = make_box(dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2);
            if (overlap_iou(a, b) > threshold) {
                suppressed[j] = 1;
            }
        }
    }
    return keep;
}

static std::vector<Detection> decode_retinaface(const ModelEntry &entry,
                                                const InferenceResult &result,
                                                const letterbox_t &letterbox,
                                                int model_w,
                                                int model_h,
                                                const std::string &image_name)
{
    std::vector<Detection> dets;
    if (result.outputs.size() < 3) {
        return dets;
    }
    const std::vector<float> &loc = result.outputs[0].values;
    const std::vector<float> &scores = result.outputs[1].values;
    int num_priors = 0;
    const float (*priors)[4] = NULL;
    if (model_h == 320) {
        num_priors = 4200;
        priors = BOX_PRIORS_320;
    } else if (model_h == 640) {
        num_priors = 16800;
        priors = BOX_PRIORS_640;
    } else {
        printf("[warn] RetinaFace decoder only supports 320/640 input, got %d\n", model_h);
        return dets;
    }
    int usable = std::min(num_priors, static_cast<int>(loc.size() / 4));
    usable = std::min(usable, static_cast<int>(scores.size() / 2));
    for (int i = 0; i < usable; ++i) {
        float score = scores[i * 2 + 1];
        if (score < entry.threshold) {
            continue;
        }
        const float v0 = 0.1f;
        const float v1 = 0.2f;
        float cx = loc[i * 4 + 0] * v0 * priors[i][2] + priors[i][0];
        float cy = loc[i * 4 + 1] * v0 * priors[i][3] + priors[i][1];
        float w = std::exp(loc[i * 4 + 2] * v1) * priors[i][2];
        float h = std::exp(loc[i * 4 + 3] * v1) * priors[i][3];
        float x1 = (cx - w * 0.5f) * model_w - letterbox.x_pad;
        float y1 = (cy - h * 0.5f) * model_h - letterbox.y_pad;
        float x2 = (cx + w * 0.5f) * model_w - letterbox.x_pad;
        float y2 = (cy + h * 0.5f) * model_h - letterbox.y_pad;
        Detection det;
        det.image = image_name;
        det.x1 = x1 / std::max(1e-6f, letterbox.scale);
        det.y1 = y1 / std::max(1e-6f, letterbox.scale);
        det.x2 = x2 / std::max(1e-6f, letterbox.scale);
        det.y2 = y2 / std::max(1e-6f, letterbox.scale);
        det.score = score;
        dets.push_back(det);
    }
    std::vector<int> keep = nms_indices(dets, 0.4f);
    std::vector<Detection> filtered;
    filtered.reserve(keep.size());
    for (size_t i = 0; i < keep.size(); ++i) {
        filtered.push_back(dets[keep[i]]);
    }
    return filtered;
}

static bool parse_box_token(const std::string &token, Box *box)
{
    std::vector<std::string> p = split_char(token, ',');
    if (p.size() != 4) {
        return false;
    }
    box->x1 = static_cast<float>(std::atof(p[0].c_str()));
    box->y1 = static_cast<float>(std::atof(p[1].c_str()));
    box->x2 = static_cast<float>(std::atof(p[2].c_str()));
    box->y2 = static_cast<float>(std::atof(p[3].c_str()));
    return true;
}

static bool load_detection_manifest(const std::string &path,
                                    std::map<std::string, std::vector<Box> > *gt)
{
    std::ifstream in(path.c_str());
    if (!in) {
        printf("[error] failed to open manifest: %s\n", path.c_str());
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> tokens = split_ws(line);
        if (tokens.size() < 2) {
            continue;
        }
        std::vector<Box> boxes;
        for (size_t i = 1; i < tokens.size();) {
            Box box;
            if (parse_box_token(tokens[i], &box)) {
                boxes.push_back(box);
                ++i;
            } else if (i + 3 < tokens.size()) {
                box.x1 = static_cast<float>(std::atof(tokens[i].c_str()));
                box.y1 = static_cast<float>(std::atof(tokens[i + 1].c_str()));
                box.x2 = static_cast<float>(std::atof(tokens[i + 2].c_str()));
                box.y2 = static_cast<float>(std::atof(tokens[i + 3].c_str()));
                boxes.push_back(box);
                i += 4;
            } else {
                break;
            }
        }
        (*gt)[tokens[0]] = boxes;
    }
    return true;
}

static double compute_ap(const std::vector<int> &tp, const std::vector<int> &fp, int total_gt)
{
    if (total_gt <= 0 || tp.empty()) {
        return 0.0;
    }
    std::vector<double> recall(tp.size());
    std::vector<double> precision(tp.size());
    int cum_tp = 0;
    int cum_fp = 0;
    for (size_t i = 0; i < tp.size(); ++i) {
        cum_tp += tp[i];
        cum_fp += fp[i];
        recall[i] = static_cast<double>(cum_tp) / total_gt;
        precision[i] = static_cast<double>(cum_tp) / std::max(1, cum_tp + cum_fp);
    }
    std::vector<double> mrec;
    std::vector<double> mpre;
    mrec.push_back(0.0);
    mpre.push_back(0.0);
    mrec.insert(mrec.end(), recall.begin(), recall.end());
    mpre.insert(mpre.end(), precision.begin(), precision.end());
    mrec.push_back(1.0);
    mpre.push_back(0.0);
    for (int i = static_cast<int>(mpre.size()) - 2; i >= 0; --i) {
        mpre[i] = std::max(mpre[i], mpre[i + 1]);
    }
    double ap = 0.0;
    for (size_t i = 1; i < mrec.size(); ++i) {
        if (mrec[i] != mrec[i - 1]) {
            ap += (mrec[i] - mrec[i - 1]) * mpre[i];
        }
    }
    return ap;
}

static bool evaluate_face_detection(const ModelEntry &entry,
                                    const CliOptions &opt,
                                    AccuracyRow *row)
{
    if (entry.output_decoder != "retinaface") {
        row->model = entry;
        row->dataset = opt.dataset;
        row->metric_summary = "unsupported decoder for detection accuracy";
        row->csv_metrics.push_back(std::make_pair("status", "unsupported_decoder"));
        return true;
    }
    std::map<std::string, std::vector<Box> > gt;
    if (!load_detection_manifest(opt.manifest_path, &gt)) {
        return false;
    }
    RknnModel model;
    if (!model.init(entry, opt.core)) {
        return false;
    }
    std::vector<Detection> predictions;
    int total_gt = 0;
    for (std::map<std::string, std::vector<Box> >::const_iterator it = gt.begin(); it != gt.end(); ++it) {
        ImageHolder image;
        if (!image.load(it->first)) {
            return false;
        }
        letterbox_t letterbox;
        std::memset(&letterbox, 0, sizeof(letterbox));
        if (!model.set_input(image.get(), &letterbox)) {
            return false;
        }
        InferenceResult result;
        if (!model.run_once(true, &result)) {
            return false;
        }
        std::vector<Detection> dets = decode_retinaface(entry, result, letterbox,
                                                        model.input_width(), model.input_height(), it->first);
        predictions.insert(predictions.end(), dets.begin(), dets.end());
        total_gt += static_cast<int>(it->second.size());
    }
    std::sort(predictions.begin(), predictions.end(), [](const Detection &a, const Detection &b) {
        return a.score > b.score;
    });

    std::map<std::string, std::vector<char> > matched;
    for (std::map<std::string, std::vector<Box> >::const_iterator it = gt.begin(); it != gt.end(); ++it) {
        matched[it->first] = std::vector<char>(it->second.size(), 0);
    }
    std::vector<int> tp(predictions.size(), 0);
    std::vector<int> fp(predictions.size(), 0);
    for (size_t i = 0; i < predictions.size(); ++i) {
        const Detection &d = predictions[i];
        std::map<std::string, std::vector<Box> >::const_iterator git = gt.find(d.image);
        float best_iou = 0.0f;
        int best_index = -1;
        if (git != gt.end()) {
            Box db = make_box(d.x1, d.y1, d.x2, d.y2);
            for (size_t j = 0; j < git->second.size(); ++j) {
                if (matched[d.image][j]) {
                    continue;
                }
                float iou = overlap_iou(db, git->second[j]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_index = static_cast<int>(j);
                }
            }
        }
        if (best_iou >= 0.5f && best_index >= 0) {
            tp[i] = 1;
            matched[d.image][best_index] = 1;
        } else {
            fp[i] = 1;
        }
    }
    int sum_tp = std::accumulate(tp.begin(), tp.end(), 0);
    int sum_fp = std::accumulate(fp.begin(), fp.end(), 0);
    double ap = compute_ap(tp, fp, total_gt);
    double recall = total_gt > 0 ? static_cast<double>(sum_tp) / total_gt : 0.0;
    double precision = (sum_tp + sum_fp) > 0 ? static_cast<double>(sum_tp) / (sum_tp + sum_fp) : 0.0;
    double fp_per_image = gt.empty() ? 0.0 : static_cast<double>(sum_fp) / gt.size();

    char summary[256];
    std::snprintf(summary, sizeof(summary),
                  "AP@0.5=%.4f recall=%.4f precision=%.4f fp/image=%.4f",
                  ap, recall, precision, fp_per_image);
    row->model = entry;
    row->dataset = opt.dataset;
    row->metric_summary = summary;
    row->csv_metrics.push_back(std::make_pair("AP@0.5", std::to_string(ap)));
    row->csv_metrics.push_back(std::make_pair("recall", std::to_string(recall)));
    row->csv_metrics.push_back(std::make_pair("precision", std::to_string(precision)));
    row->csv_metrics.push_back(std::make_pair("fp_per_image", std::to_string(fp_per_image)));
    if (opt.dataset == "wider") {
        row->csv_metrics.push_back(std::make_pair("wider_easy", "requires_official_eval"));
        row->csv_metrics.push_back(std::make_pair("wider_medium", "requires_official_eval"));
        row->csv_metrics.push_back(std::make_pair("wider_hard", "requires_official_eval"));
    }
    return true;
}

static double output_score(const ModelEntry &entry, const InferenceResult &result)
{
    if (result.outputs.empty()) {
        return 0.0;
    }
    int index = std::max(0, std::min(entry.output_index, static_cast<int>(result.outputs.size()) - 1));
    const std::vector<float> &v = result.outputs[index].values;
    if (v.empty()) {
        return 0.0;
    }
    int pos = std::max(0, std::min(entry.positive_index, static_cast<int>(v.size()) - 1));
    if (entry.output_activation == "softmax" && v.size() >= 2) {
        double max_v = *std::max_element(v.begin(), v.end());
        double denom = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            denom += std::exp(v[i] - max_v);
        }
        return denom > 0.0 ? std::exp(v[pos] - max_v) / denom : 0.0;
    }
    if (entry.output_activation == "sigmoid") {
        return 1.0 / (1.0 + std::exp(-v[pos]));
    }
    return v[pos];
}

static bool parse_binary_label(const std::string &label, int *out)
{
    std::string l = lower(label);
    if (l == "1" || l == "live" || l == "real" || l == "bonafide" ||
        l == "bona_fide" || l == "same" || l == "positive" || l == "match") {
        *out = 1;
        return true;
    }
    if (l == "0" || l == "spoof" || l == "attack" || l == "fake" ||
        l == "different" || l == "diff" || l == "negative" || l == "nonmatch") {
        *out = 0;
        return true;
    }
    return false;
}

static double compute_auc(std::vector<std::pair<double, int> > scores)
{
    int pos = 0;
    int neg = 0;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i].second) {
            pos++;
        } else {
            neg++;
        }
    }
    if (pos == 0 || neg == 0) {
        return 0.0;
    }
    std::sort(scores.begin(), scores.end());
    double rank_sum = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i].second) {
            rank_sum += i + 1;
        }
    }
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg);
}

static double compute_eer(std::vector<std::pair<double, int> > scores)
{
    if (scores.empty()) {
        return 0.0;
    }
    int positives = 0;
    int negatives = 0;
    for (size_t i = 0; i < scores.size(); ++i) {
        positives += scores[i].second ? 1 : 0;
        negatives += scores[i].second ? 0 : 1;
    }
    if (positives == 0 || negatives == 0) {
        return 0.0;
    }
    std::sort(scores.begin(), scores.end(), [](const std::pair<double, int> &a,
                                               const std::pair<double, int> &b) {
        return a.first > b.first;
    });
    int tp = 0;
    int fp = 0;
    double best_diff = 1e9;
    double best_eer = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i].second) {
            tp++;
        } else {
            fp++;
        }
        double fpr = static_cast<double>(fp) / negatives;
        double fnr = static_cast<double>(positives - tp) / positives;
        double diff = std::fabs(fpr - fnr);
        if (diff < best_diff) {
            best_diff = diff;
            best_eer = (fpr + fnr) * 0.5;
        }
    }
    return best_eer;
}

static bool evaluate_liveness(const ModelEntry &entry,
                              const CliOptions &opt,
                              AccuracyRow *row)
{
    std::ifstream in(opt.manifest_path.c_str());
    if (!in) {
        printf("[error] failed to open manifest: %s\n", opt.manifest_path.c_str());
        return false;
    }
    RknnModel model;
    if (!model.init(entry, opt.core)) {
        return false;
    }
    int live_total = 0;
    int spoof_total = 0;
    int live_pred_spoof = 0;
    int spoof_pred_live = 0;
    std::vector<std::pair<double, int> > scores;
    std::string line;
    while (std::getline(in, line)) {
        size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> tokens = split_ws(line);
        if (tokens.size() < 2) {
            continue;
        }
        int label = 0;
        if (!parse_binary_label(tokens[1], &label)) {
            printf("[warn] skipping invalid liveness label: %s\n", line.c_str());
            continue;
        }
        ImageHolder image;
        if (!image.load(tokens[0])) {
            return false;
        }
        letterbox_t letterbox;
        std::memset(&letterbox, 0, sizeof(letterbox));
        if (!model.set_input(image.get(), &letterbox)) {
            return false;
        }
        InferenceResult result;
        if (!model.run_once(true, &result)) {
            return false;
        }
        double score_live = output_score(entry, result);
        int pred_live = score_live >= entry.threshold ? 1 : 0;
        scores.push_back(std::make_pair(score_live, label));
        if (label) {
            live_total++;
            if (!pred_live) {
                live_pred_spoof++;
            }
        } else {
            spoof_total++;
            if (pred_live) {
                spoof_pred_live++;
            }
        }
    }
    double bpcer = live_total > 0 ? static_cast<double>(live_pred_spoof) / live_total : 0.0;
    double apcer = spoof_total > 0 ? static_cast<double>(spoof_pred_live) / spoof_total : 0.0;
    double acer = (apcer + bpcer) * 0.5;
    double auc = compute_auc(scores);
    double eer = compute_eer(scores);

    char summary[256];
    std::snprintf(summary, sizeof(summary),
                  "APCER=%.4f BPCER=%.4f ACER=%.4f AUC=%.4f EER=%.4f",
                  apcer, bpcer, acer, auc, eer);
    row->model = entry;
    row->dataset = opt.dataset;
    row->metric_summary = summary;
    row->csv_metrics.push_back(std::make_pair("APCER", std::to_string(apcer)));
    row->csv_metrics.push_back(std::make_pair("BPCER", std::to_string(bpcer)));
    row->csv_metrics.push_back(std::make_pair("ACER", std::to_string(acer)));
    row->csv_metrics.push_back(std::make_pair("AUC", std::to_string(auc)));
    row->csv_metrics.push_back(std::make_pair("EER", std::to_string(eer)));
    return true;
}

static std::vector<float> extract_embedding_values(const ModelEntry &entry,
                                                   const InferenceResult &result)
{
    std::vector<float> out;
    if (result.outputs.empty()) {
        return out;
    }
    int index = std::max(0, std::min(entry.output_index, static_cast<int>(result.outputs.size()) - 1));
    out = result.outputs[index].values;
    if (entry.feature_norm == "l2") {
        double norm = 0.0;
        for (size_t i = 0; i < out.size(); ++i) {
            norm += out[i] * out[i];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = static_cast<float>(out[i] / norm);
            }
        }
    }
    return out;
}

static double cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
{
    size_t n = std::min(a.size(), b.size());
    if (n == 0) {
        return 0.0;
    }
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-12 ? dot / denom : 0.0;
}

static bool embed_image(RknnModel *model,
                        const ModelEntry &entry,
                        const std::string &path,
                        std::map<std::string, std::vector<float> > *cache,
                        std::vector<float> *embedding)
{
    std::map<std::string, std::vector<float> >::const_iterator cached = cache->find(path);
    if (cached != cache->end()) {
        *embedding = cached->second;
        return true;
    }
    ImageHolder image;
    if (!image.load(path)) {
        return false;
    }
    letterbox_t letterbox;
    std::memset(&letterbox, 0, sizeof(letterbox));
    if (!model->set_input(image.get(), &letterbox)) {
        return false;
    }
    InferenceResult result;
    if (!model->run_once(true, &result)) {
        return false;
    }
    *embedding = extract_embedding_values(entry, result);
    (*cache)[path] = *embedding;
    return true;
}

struct PairScore {
    double score = 0.0;
    int same = 0;
    int fold = 0;
};

static double accuracy_at_threshold(const std::vector<PairScore> &pairs,
                                    const std::vector<int> &indices,
                                    double threshold)
{
    if (indices.empty()) {
        return 0.0;
    }
    int correct = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        const PairScore &p = pairs[indices[i]];
        int pred = p.score >= threshold ? 1 : 0;
        if (pred == p.same) {
            correct++;
        }
    }
    return static_cast<double>(correct) / indices.size();
}

static double best_threshold(const std::vector<PairScore> &pairs,
                             const std::vector<int> &indices)
{
    if (indices.empty()) {
        return 0.0;
    }
    struct ScoreLabel {
        double score;
        int same;
    };
    std::vector<ScoreLabel> sorted;
    sorted.reserve(indices.size());
    int positives = 0;
    int negatives = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        const PairScore &p = pairs[indices[i]];
        sorted.push_back({p.score, p.same});
        if (p.same) {
            positives++;
        } else {
            negatives++;
        }
    }
    std::sort(sorted.begin(), sorted.end(), [](const ScoreLabel &a, const ScoreLabel &b) {
        return a.score > b.score;
    });

    double best_t = sorted[0].score;
    double best_acc = -1.0;
    int accepted_pos = 0;
    int accepted_neg = 0;
    size_t i = 0;
    while (i < sorted.size()) {
        double threshold = sorted[i].score;
        while (i < sorted.size() && sorted[i].score == threshold) {
            if (sorted[i].same) {
                accepted_pos++;
            } else {
                accepted_neg++;
            }
            i++;
        }
        int correct = accepted_pos + (negatives - accepted_neg);
        double acc = static_cast<double>(correct) / sorted.size();
        if (acc > best_acc) {
            best_acc = acc;
            best_t = threshold;
        }
    }
    return best_t;
}

struct TarAtFarResult {
    double tar = 0.0;
    double threshold = 0.0;
};

static TarAtFarResult tar_at_far(std::vector<PairScore> pairs, double far)
{
    TarAtFarResult result;
    std::vector<double> neg_scores;
    int positives = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].same) {
            positives++;
        } else {
            neg_scores.push_back(pairs[i].score);
        }
    }
    if (positives == 0 || neg_scores.empty()) {
        return result;
    }
    std::sort(neg_scores.begin(), neg_scores.end(), std::greater<double>());
    size_t allowed_fp = static_cast<size_t>(std::floor(far * neg_scores.size()));
    size_t threshold_index = std::min(allowed_fp, neg_scores.size() - 1);
    result.threshold = neg_scores[threshold_index];
    int true_accept = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].same && pairs[i].score >= result.threshold) {
            true_accept++;
        }
    }
    result.tar = static_cast<double>(true_accept) / positives;
    return result;
}

static bool evaluate_feature_extraction(const ModelEntry &entry,
                                        const CliOptions &opt,
                                        AccuracyRow *row)
{
    std::ifstream in(opt.manifest_path.c_str());
    if (!in) {
        printf("[error] failed to open manifest: %s\n", opt.manifest_path.c_str());
        return false;
    }
    RknnModel model;
    if (!model.init(entry, opt.core)) {
        return false;
    }
    std::map<std::string, std::vector<float> > cache;
    std::vector<PairScore> pairs;
    std::string line;
    int auto_fold = 0;
    while (std::getline(in, line)) {
        size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> tokens = split_ws(line);
        if (tokens.size() < 3) {
            continue;
        }
        int same = 0;
        if (!parse_binary_label(tokens[2], &same)) {
            printf("[warn] skipping invalid pair label: %s\n", line.c_str());
            continue;
        }
        int fold = tokens.size() >= 4 ? std::atoi(tokens[3].c_str()) : (auto_fold++ % 10);
        fold = std::max(0, std::min(9, fold));
        std::vector<float> e1;
        std::vector<float> e2;
        if (!embed_image(&model, entry, tokens[0], &cache, &e1) ||
            !embed_image(&model, entry, tokens[1], &cache, &e2)) {
            return false;
        }
        PairScore p;
        p.score = cosine_similarity(e1, e2);
        p.same = same;
        p.fold = fold;
        pairs.push_back(p);
    }
    if (pairs.empty()) {
        printf("[error] no valid feature pairs in manifest\n");
        return false;
    }

    double acc_sum = 0.0;
    double threshold_sum = 0.0;
    int used_folds = 0;
    for (int fold = 0; fold < 10; ++fold) {
        std::vector<int> train;
        std::vector<int> test;
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (pairs[i].fold == fold) {
                test.push_back(static_cast<int>(i));
            } else {
                train.push_back(static_cast<int>(i));
            }
        }
        if (test.empty()) {
            continue;
        }
        double t = train.empty() ? best_threshold(pairs, test) : best_threshold(pairs, train);
        double acc = accuracy_at_threshold(pairs, test, t);
        acc_sum += acc;
        threshold_sum += t;
        used_folds++;
    }
    double mean_acc = used_folds > 0 ? acc_sum / used_folds : 0.0;
    double mean_threshold = used_folds > 0 ? threshold_sum / used_folds : 0.0;
    std::vector<std::pair<double, int> > scores;
    for (size_t i = 0; i < pairs.size(); ++i) {
        scores.push_back(std::make_pair(pairs[i].score, pairs[i].same));
    }
    double eer = compute_eer(scores);
    TarAtFarResult tar1e3 = tar_at_far(pairs, 1e-3);
    TarAtFarResult tar1e4 = tar_at_far(pairs, 1e-4);

    char summary[512];
    std::snprintf(summary, sizeof(summary),
                  "10fold_acc=%.4f EER=%.4f threshold=%.4f "
                  "TAR@FAR1e-3=%.4f threshold@FAR1e-3=%.4f "
                  "TAR@FAR1e-4=%.4f threshold@FAR1e-4=%.4f",
                  mean_acc, eer, mean_threshold,
                  tar1e3.tar, tar1e3.threshold,
                  tar1e4.tar, tar1e4.threshold);
    row->model = entry;
    row->dataset = opt.dataset;
    row->metric_summary = summary;
    row->csv_metrics.push_back(std::make_pair("10fold_acc", std::to_string(mean_acc)));
    row->csv_metrics.push_back(std::make_pair("EER", std::to_string(eer)));
    row->csv_metrics.push_back(std::make_pair("threshold", std::to_string(mean_threshold)));
    row->csv_metrics.push_back(std::make_pair("TAR@FAR1e-3", std::to_string(tar1e3.tar)));
    row->csv_metrics.push_back(std::make_pair("threshold@FAR1e-3", std::to_string(tar1e3.threshold)));
    row->csv_metrics.push_back(std::make_pair("TAR@FAR1e-4", std::to_string(tar1e4.tar)));
    row->csv_metrics.push_back(std::make_pair("threshold@FAR1e-4", std::to_string(tar1e4.threshold)));
    return true;
}

static bool evaluate_accuracy(const ModelEntry &entry,
                              const CliOptions &opt,
                              AccuracyRow *row)
{
    if (entry.task == "face_detection") {
        return evaluate_face_detection(entry, opt, row);
    }
    if (entry.task == "liveness") {
        return evaluate_liveness(entry, opt, row);
    }
    if (entry.task == "feature_extraction") {
        return evaluate_feature_extraction(entry, opt, row);
    }
    row->model = entry;
    row->dataset = opt.dataset;
    row->metric_summary = "unsupported task";
    row->csv_metrics.push_back(std::make_pair("status", "unsupported_task"));
    return true;
}

static void print_speed_table(const std::vector<SpeedRow> &rows)
{
    printf("\nSpeed benchmark, RK3588 single-core, headline latency = rknn_run + rknn_outputs_get\n");
    printf("| classification | demo | model_name | input_shape | dtype | RK3588@single_core (avg/p50/p95) | fps (avg/p50/p95) |\n");
    printf("|---|---|---|---|---|---:|---:|\n");
    for (size_t i = 0; i < rows.size(); ++i) {
        const SpeedRow &r = rows[i];
        double fps_avg = r.ms.avg > 0.0 ? 1000.0 / r.ms.avg : 0.0;
        double fps_p50 = r.ms.p50 > 0.0 ? 1000.0 / r.ms.p50 : 0.0;
        double fps_p95 = r.ms.p95 > 0.0 ? 1000.0 / r.ms.p95 : 0.0;
        printf("| %s | %s | %s | `%s` | %s | %.2f/%.2f/%.2f ms | %.2f/%.2f/%.2f |\n",
               r.model.classification.c_str(),
               r.model.demo.c_str(),
               r.model.name.c_str(),
               tensor_shape_string(r.model.input_shape).c_str(),
               r.model.dtype.c_str(),
               r.ms.avg, r.ms.p50, r.ms.p95,
               fps_avg, fps_p50, fps_p95);
    }
}

static void print_accuracy_table(const std::vector<AccuracyRow> &rows)
{
    printf("\nAccuracy benchmark\n");
    printf("| classification | task | demo | model_name | dataset | metrics |\n");
    printf("|---|---|---|---|---|---|\n");
    for (size_t i = 0; i < rows.size(); ++i) {
        const AccuracyRow &r = rows[i];
        printf("| %s | %s | %s | %s | %s | %s |\n",
               r.model.classification.c_str(),
               r.model.task.c_str(),
               r.model.demo.c_str(),
               r.model.name.c_str(),
               r.dataset.c_str(),
               r.metric_summary.c_str());
    }
}

static std::string csv_escape(const std::string &value)
{
    if (value.find_first_of(",\"\n") == std::string::npos) {
        return value;
    }
    std::string out = "\"";
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '"') {
            out += "\"\"";
        } else {
            out += value[i];
        }
    }
    out += "\"";
    return out;
}

static void write_csv(const std::string &path,
                      const std::vector<SpeedRow> &speed_rows,
                      const std::vector<AccuracyRow> &accuracy_rows)
{
    if (path.empty()) {
        return;
    }
    std::ofstream out(path.c_str());
    if (!out) {
        printf("[error] failed to write CSV: %s\n", path.c_str());
        return;
    }
    out << "section,classification,task,demo,model_name,input_shape,dtype,dataset,metric,value\n";
    for (size_t i = 0; i < speed_rows.size(); ++i) {
        const SpeedRow &r = speed_rows[i];
        out << "speed," << csv_escape(r.model.classification) << ","
            << csv_escape(r.model.task) << "," << csv_escape(r.model.demo) << ","
            << csv_escape(r.model.name) << "," << csv_escape(tensor_shape_string(r.model.input_shape)) << ","
            << csv_escape(r.model.dtype) << ",,"
            << "latency_avg_ms," << r.ms.avg << "\n";
        out << "speed," << csv_escape(r.model.classification) << ","
            << csv_escape(r.model.task) << "," << csv_escape(r.model.demo) << ","
            << csv_escape(r.model.name) << "," << csv_escape(tensor_shape_string(r.model.input_shape)) << ","
            << csv_escape(r.model.dtype) << ",,"
            << "latency_p50_ms," << r.ms.p50 << "\n";
        out << "speed," << csv_escape(r.model.classification) << ","
            << csv_escape(r.model.task) << "," << csv_escape(r.model.demo) << ","
            << csv_escape(r.model.name) << "," << csv_escape(tensor_shape_string(r.model.input_shape)) << ","
            << csv_escape(r.model.dtype) << ",,"
            << "latency_p95_ms," << r.ms.p95 << "\n";
    }
    for (size_t i = 0; i < accuracy_rows.size(); ++i) {
        const AccuracyRow &r = accuracy_rows[i];
        for (size_t j = 0; j < r.csv_metrics.size(); ++j) {
            out << "accuracy," << csv_escape(r.model.classification) << ","
                << csv_escape(r.model.task) << "," << csv_escape(r.model.demo) << ","
                << csv_escape(r.model.name) << "," << csv_escape(tensor_shape_string(r.model.input_shape)) << ","
                << csv_escape(r.model.dtype) << "," << csv_escape(r.dataset) << ","
                << csv_escape(r.csv_metrics[j].first) << "," << csv_escape(r.csv_metrics[j].second) << "\n";
        }
    }
    printf("[csv] wrote %s\n", path.c_str());
}

static bool model_selected(const ModelEntry &entry, const CliOptions &opt)
{
    return opt.task == "all" || entry.task == opt.task;
}

int main(int argc, char **argv)
{
    CliOptions opt;
    if (!parse_cli(argc, argv, &opt)) {
        print_usage(argv[0]);
        return -1;
    }

    std::vector<ModelEntry> models;
    if (!parse_model_registry(opt.models_path, &models)) {
        return -1;
    }
    if (models.empty()) {
        printf("[error] no models found in registry: %s\n", opt.models_path.c_str());
        return -1;
    }

    printf("[config] models=%s task=%s core=%d warmup=%d iters=%d speed=%d accuracy=%d\n",
           opt.models_path.c_str(), opt.task.c_str(), opt.core, opt.warmup, opt.iters,
           opt.run_speed ? 1 : 0, opt.run_accuracy ? 1 : 0);
    printf("[config] RKNN perf collection is intentionally disabled; no RKNN_FLAG_COLLECT_PERF_MASK is used.\n");

    std::vector<SpeedRow> speed_rows;
    std::vector<AccuracyRow> accuracy_rows;
    for (size_t i = 0; i < models.size(); ++i) {
        const ModelEntry &entry = models[i];
        if (!model_selected(entry, opt)) {
            continue;
        }
        if (!file_exists(entry.resolved_path)) {
            printf("[warn] skip missing model %s path=%s\n",
                   entry.name.c_str(), entry.resolved_path.c_str());
            continue;
        }
        if (opt.run_speed) {
            printf("[speed] %s/%s\n", entry.demo.c_str(), entry.name.c_str());
            SpeedRow row;
            if (!run_speed_benchmark(entry, opt, &row)) {
                return -1;
            }
            speed_rows.push_back(row);
        }
        if (opt.run_accuracy) {
            printf("[accuracy] %s/%s task=%s dataset=%s\n",
                   entry.demo.c_str(), entry.name.c_str(), entry.task.c_str(), opt.dataset.c_str());
            AccuracyRow row;
            if (!evaluate_accuracy(entry, opt, &row)) {
                return -1;
            }
            accuracy_rows.push_back(row);
        }
    }

    if (!speed_rows.empty()) {
        print_speed_table(speed_rows);
    }
    if (!accuracy_rows.empty()) {
        print_accuracy_table(accuracy_rows);
    }
    write_csv(opt.output_path, speed_rows, accuracy_rows);
    if (speed_rows.empty() && accuracy_rows.empty()) {
        printf("[warn] no benchmark rows produced. Check --task and model paths.\n");
    }
    return 0;
}
