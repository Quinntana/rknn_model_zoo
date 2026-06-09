// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <sys/time.h>
#include <string>
#include <algorithm> // Needed for std::max / std::min
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <vector>
#include <thread>

// Include OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

// Include Both Models
#include "yolo11.h"
#include "retinaface.h"
#include "image_utils.h"
#include "file_utils.h"

double get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static const char *LOG_RESET = "\033[0m";
static const char *LOG_RED = "\033[31m";
static const char *LOG_GREEN = "\033[32m";
static const char *LOG_YELLOW = "\033[33m";
static const char *LOG_BLUE = "\033[34m";
static const char *LOG_CYAN = "\033[36m";

static void log_stage_time(const char *color, const char *stage, size_t seq, int worker_id, int ctx_id, double ms)
{
    printf("%s[%s][worker-%d][ctx-%d][frame-%zu] %.2f ms%s\n", color, stage, worker_id, ctx_id, seq, ms, LOG_RESET);
}

struct AppConfig {
    std::string source = "rtsp";
    std::string rtsp_url = "rtsp://admin:admin@192.168.1.156:554/stream1";
    std::string rtsp_transport = "tcp";
    int rtsp_latency_ms = 200;
    std::string camera_codec = "h264";
    int udp_port = 5000;
    size_t clip_frames = 150;
    size_t max_buffered_frames = 300;
    double output_fps = 25.0;
    std::string output_path = "output_cascaded.mp4";
    int yolo_worker_count = 1;
    int retina_worker_count = 1;
    size_t max_yolo_queue_size = 1;
    size_t max_retina_queue_size = 2;
    int yolo_interval = 1;
    int min_retina_crop_size = 64;
    int max_retina_crops = 4;
    int videoconvert_threads = 4;
    int retina_reinfer_interval = 1;
    std::string npu_core_mode = "balanced";
    bool rknn_perf = false;
    bool async_retina = false;
};

struct FramePacket {
    size_t seq = 0;
    cv::Mat bgr_frame;
    cv::Mat rgb_frame;
    object_detect_result_list od_results;
    double processing_start_ms = 0.0;
    bool has_yolo = false;
    bool has_retina = false;
    bool run_retina = true;
    bool record_on_retina = true;
    bool dropped = false;
};

struct RecordedFrame {
    size_t seq = 0;
    cv::Mat bgr_frame;
};

struct RuntimeStats {
    std::atomic<size_t> yolo_processed{0};
    std::atomic<size_t> processed_frames{0};
    std::atomic<size_t> yolo_queue_drops{0};
    std::atomic<size_t> retina_queue_drops{0};
    std::atomic<size_t> interval_reuse_frames{0};
    std::atomic<size_t> retina_interval_skips{0};
    std::atomic<size_t> async_retina_outputs{0};
    std::atomic<size_t> retina_crops_run{0};
    std::atomic<size_t> retina_crops_skipped_small{0};
    std::atomic<size_t> retina_crops_skipped_limit{0};
};

struct TimingSnapshot {
    size_t count = 0;
    double avg_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

class RollingTiming {
public:
    explicit RollingTiming(size_t max_samples = 256)
        : max_samples_(max_samples)
    {
    }

    void add(double ms)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        samples_.push_back(ms);
        while (samples_.size() > max_samples_) {
            samples_.pop_front();
        }
    }

    TimingSnapshot snapshot()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        TimingSnapshot snap;
        snap.count = samples_.size();
        if (samples_.empty()) {
            return snap;
        }

        std::vector<double> sorted(samples_.begin(), samples_.end());
        std::sort(sorted.begin(), sorted.end());
        double total = 0.0;
        for (double sample : sorted) {
            total += sample;
        }

        snap.avg_ms = total / sorted.size();
        snap.p50_ms = percentile(sorted, 0.50);
        snap.p95_ms = percentile(sorted, 0.95);
        return snap;
    }

private:
    static double percentile(const std::vector<double> &sorted, double pct)
    {
        if (sorted.empty()) {
            return 0.0;
        }
        size_t index = (size_t)((sorted.size() - 1) * pct + 0.5);
        index = std::min(index, sorted.size() - 1);
        return sorted[index];
    }

    std::mutex mutex_;
    std::deque<double> samples_;
    size_t max_samples_ = 0;
};

class DetectionCache {
public:
    void update(const object_detect_result_list &od_results)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        od_results_ = od_results;
        valid_ = true;
    }

    bool snapshot(object_detect_result_list *od_results)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid_) {
            return false;
        }
        *od_results = od_results_;
        return true;
    }

private:
    std::mutex mutex_;
    object_detect_result_list od_results_;
    bool valid_ = false;
};

static void draw_yolo_results(cv::Mat *bgr_frame, const object_detect_result_list &od_results)
{
    for (int i = 0; i < od_results.count; i++) {
        const object_detect_result *det_result = &(od_results.results[i]);

        cv::rectangle(*bgr_frame,
                      cv::Point(det_result->box.left, det_result->box.top),
                      cv::Point(det_result->box.right, det_result->box.bottom),
                      cv::Scalar(0, 255, 0), 2);
        char text[256];
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        cv::putText(*bgr_frame, text,
                    cv::Point(det_result->box.left, det_result->box.top - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
}

class RecorderBuffer {
public:
    RecorderBuffer(size_t max_buffered_frames, size_t max_reorder_pending)
        : max_buffered_frames_(max_buffered_frames),
          max_reorder_pending_(max_reorder_pending)
    {
    }

    void add_completed(size_t seq, cv::Mat bgr_frame)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (seq < next_seq_) {
            return;
        }
        pending_[seq] = RecordedFrame{seq, std::move(bgr_frame)};
        flush_locked();
    }

    void mark_dropped(size_t seq)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (seq < next_seq_) {
            return;
        }
        dropped_.insert(seq);
        flush_locked();
    }

    std::vector<RecordedFrame> snapshot_last(size_t frame_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<RecordedFrame> frames;
        if (ring_.empty()) {
            return frames;
        }

        size_t copy_count = std::min(frame_count, ring_.size());
        frames.reserve(copy_count);
        size_t start = ring_.size() - copy_count;
        for (size_t i = start; i < ring_.size(); ++i) {
            frames.push_back(RecordedFrame{ring_[i].seq, ring_[i].bgr_frame.clone()});
        }
        return frames;
    }

    size_t buffered_count()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return ring_.size();
    }

private:
    void flush_locked()
    {
        while (true) {
            auto pending_it = pending_.find(next_seq_);
            if (pending_it != pending_.end()) {
                ring_.push_back(std::move(pending_it->second));
                pending_.erase(pending_it);
                next_seq_++;
                while (ring_.size() > max_buffered_frames_) {
                    ring_.pop_front();
                }
                continue;
            }

            auto dropped_it = dropped_.find(next_seq_);
            if (dropped_it != dropped_.end()) {
                dropped_.erase(dropped_it);
                next_seq_++;
                continue;
            }

            if (pending_.size() > max_reorder_pending_ && !pending_.empty()) {
                size_t lowest_pending_seq = pending_.begin()->first;
                printf("%s[recorder][reorder] skipping missing frames %zu-%zu to avoid reorder stall%s\n",
                       LOG_YELLOW, next_seq_, lowest_pending_seq - 1, LOG_RESET);
                next_seq_ = lowest_pending_seq;
                continue;
            }

            break;
        }
    }

    std::mutex mutex_;
    std::map<size_t, RecordedFrame> pending_;
    std::set<size_t> dropped_;
    std::deque<RecordedFrame> ring_;
    size_t next_seq_ = 0;
    size_t max_buffered_frames_ = 0;
    size_t max_reorder_pending_ = 0;
};

struct FaceLandmarks {
    cv::Point points[5];
};

class RetinaOverlayCache {
public:
    void update(std::vector<FaceLandmarks> faces)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        faces_ = std::move(faces);
    }

    void draw(cv::Mat *bgr_frame)
    {
        std::vector<FaceLandmarks> faces;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            faces = faces_;
        }

        for (const FaceLandmarks &face : faces) {
            for (int p = 0; p < 5; ++p) {
                cv::circle(*bgr_frame, face.points[p], 4, cv::Scalar(0, 165, 255), -1);
            }
        }
    }

private:
    std::mutex mutex_;
    std::vector<FaceLandmarks> faces_;
};

class TerminalModeGuard {
public:
    TerminalModeGuard()
    {
        enabled_ = isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO, &old_term_) == 0;
        if (!enabled_) {
            return;
        }

        struct termios new_term = old_term_;
        new_term.c_lflag &= ~(ICANON | ECHO);
        new_term.c_cc[VMIN] = 0;
        new_term.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &new_term) != 0) {
            enabled_ = false;
        }
    }

    ~TerminalModeGuard()
    {
        restore();
    }

    void restore()
    {
        if (enabled_) {
            tcsetattr(STDIN_FILENO, TCSANOW, &old_term_);
            enabled_ = false;
        }
    }

    bool enabled() const
    {
        return enabled_;
    }

private:
    bool enabled_ = false;
    struct termios old_term_;
};

static bool parse_int_arg(const char *value, int *out)
{
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out = (int)parsed;
    return true;
}

static bool parse_size_arg(const char *value, size_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out = (size_t)parsed;
    return true;
}

static bool parse_double_arg(const char *value, double *out)
{
    char *end = NULL;
    errno = 0;
    double parsed = strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out = parsed;
    return true;
}

static void print_usage(const char *program)
{
    printf("Usage: %s <yolo_model> <retina_model> [--source rtsp|udp] [--rtsp-url URL] [--rtsp-transport tcp|udp] [--rtsp-latency-ms 200] [--camera-codec h264|h265] [--udp-port 5000] [--clip-frames 150] [--max-buffered-frames 300] [--fps 25] [--output output_cascaded.mp4] [--yolo-workers 1] [--retina-workers 1] [--max-yolo-queue 1] [--max-retina-queue 2] [--yolo-interval 1] [--retina-reinfer-interval 1] [--min-retina-crop 64] [--max-retina-crops 4] [--videoconvert-threads 4] [--npu-core-mode balanced|yolo-all|round-robin] [--rknn-perf] [--async-retina]\n", program);
}

static bool parse_config(int argc, char **argv, AppConfig *config)
{
    if (argc < 3) {
        return false;
    }

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
            config->source = argv[++i];
        } else if (strcmp(argv[i], "--rtsp-url") == 0 && i + 1 < argc) {
            config->rtsp_url = argv[++i];
            config->source = "rtsp";
        } else if (strcmp(argv[i], "--rtsp-transport") == 0 && i + 1 < argc) {
            config->rtsp_transport = argv[++i];
            config->source = "rtsp";
        } else if (strcmp(argv[i], "--rtsp-latency-ms") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->rtsp_latency_ms)) {
                return false;
            }
            config->source = "rtsp";
        } else if (strcmp(argv[i], "--camera-codec") == 0 && i + 1 < argc) {
            config->camera_codec = argv[++i];
        } else if (strcmp(argv[i], "--udp-port") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->udp_port)) {
                return false;
            }
            config->source = "udp";
        } else if (strcmp(argv[i], "--clip-frames") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->clip_frames)) {
                return false;
            }
        } else if (strcmp(argv[i], "--max-buffered-frames") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->max_buffered_frames)) {
                return false;
            }
        } else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            if (!parse_double_arg(argv[++i], &config->output_fps)) {
                return false;
            }
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            config->output_path = argv[++i];
        } else if (strcmp(argv[i], "--yolo-workers") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->yolo_worker_count)) {
                return false;
            }
        } else if (strcmp(argv[i], "--retina-workers") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->retina_worker_count)) {
                return false;
            }
        } else if (strcmp(argv[i], "--max-yolo-queue") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->max_yolo_queue_size)) {
                return false;
            }
        } else if (strcmp(argv[i], "--max-retina-queue") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->max_retina_queue_size)) {
                return false;
            }
        } else if (strcmp(argv[i], "--yolo-interval") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->yolo_interval)) {
                return false;
            }
        } else if (strcmp(argv[i], "--retina-reinfer-interval") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->retina_reinfer_interval)) {
                return false;
            }
        } else if (strcmp(argv[i], "--min-retina-crop") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->min_retina_crop_size)) {
                return false;
            }
        } else if (strcmp(argv[i], "--max-retina-crops") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->max_retina_crops)) {
                return false;
            }
        } else if (strcmp(argv[i], "--videoconvert-threads") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->videoconvert_threads)) {
                return false;
            }
        } else if (strcmp(argv[i], "--npu-core-mode") == 0 && i + 1 < argc) {
            config->npu_core_mode = argv[++i];
        } else if (strcmp(argv[i], "--rknn-perf") == 0) {
            config->rknn_perf = true;
        } else if (strcmp(argv[i], "--async-retina") == 0) {
            config->async_retina = true;
        } else {
            return false;
        }
    }

    if (config->source != "rtsp" && config->source != "udp") {
        printf("--source must be 'rtsp' or 'udp'\n");
        return false;
    }
    if (config->source == "rtsp" && config->rtsp_url.empty()) {
        printf("--rtsp-url must not be empty when --source rtsp is used\n");
        return false;
    }
    if (config->rtsp_transport != "tcp" && config->rtsp_transport != "udp") {
        printf("--rtsp-transport must be 'tcp' or 'udp'\n");
        return false;
    }
    if (config->rtsp_latency_ms < 0) {
        printf("--rtsp-latency-ms must be >= 0\n");
        return false;
    }
    if (config->camera_codec != "h264" && config->camera_codec != "h265") {
        printf("--camera-codec must be 'h264' or 'h265'\n");
        return false;
    }
    if (config->udp_port <= 0 || config->udp_port > 65535) {
        printf("Invalid --udp-port: %d\n", config->udp_port);
        return false;
    }
    if (config->clip_frames == 0) {
        printf("--clip-frames must be greater than 0\n");
        return false;
    }
    if (config->max_buffered_frames == 0) {
        printf("--max-buffered-frames must be greater than 0\n");
        return false;
    }
    if (config->clip_frames > config->max_buffered_frames) {
        printf("--clip-frames (%zu) must be <= --max-buffered-frames (%zu)\n",
               config->clip_frames, config->max_buffered_frames);
        return false;
    }
    if (config->output_fps <= 0.0) {
        printf("--fps must be greater than 0\n");
        return false;
    }
    if (config->output_path.empty()) {
        printf("--output must not be empty\n");
        return false;
    }
    if (config->yolo_worker_count <= 0) {
        printf("--yolo-workers must be greater than 0\n");
        return false;
    }
    if (config->retina_worker_count <= 0) {
        printf("--retina-workers must be greater than 0\n");
        return false;
    }
    if (config->max_yolo_queue_size == 0) {
        printf("--max-yolo-queue must be greater than 0\n");
        return false;
    }
    if (config->max_retina_queue_size == 0) {
        printf("--max-retina-queue must be greater than 0\n");
        return false;
    }
    if (config->yolo_interval <= 0) {
        printf("--yolo-interval must be greater than 0\n");
        return false;
    }
    if (config->retina_reinfer_interval <= 0) {
        printf("--retina-reinfer-interval must be greater than 0\n");
        return false;
    }
    if (config->min_retina_crop_size < 0) {
        printf("--min-retina-crop must be >= 0\n");
        return false;
    }
    if (config->max_retina_crops < 0) {
        printf("--max-retina-crops must be >= 0\n");
        return false;
    }
    if (config->videoconvert_threads <= 0) {
        printf("--videoconvert-threads must be greater than 0\n");
        return false;
    }
    if (config->npu_core_mode != "balanced" &&
        config->npu_core_mode != "yolo-all" &&
        config->npu_core_mode != "round-robin") {
        printf("--npu-core-mode must be 'balanced', 'yolo-all', or 'round-robin'\n");
        return false;
    }
    return true;
}

static std::string decode_output_queue()
{
    return " ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0";
}

static std::string bgr_convert(const AppConfig &config)
{
    return " ! videoconvert n-threads=" + std::to_string(config.videoconvert_threads) +
           " dither=none ! video/x-raw,format=BGR";
}

static std::string build_input_pipeline(const AppConfig &config)
{
    std::string depay = "rtph264depay";
    std::string parse = "h264parse";
    if (config.camera_codec == "h265") {
        depay = "rtph265depay";
        parse = "h265parse";
    }

    if (config.source == "udp") {
        return "udpsrc port=" + std::to_string(config.udp_port) +
               " caps=\"video/mpegts, systemstream=(boolean)true, packetsize=(int)188\""
               " ! tsdemux"
               " ! " + parse +
               " ! mppvideodec" + decode_output_queue() +
               bgr_convert(config) +
               " ! appsink drop=true max-buffers=1 sync=false";
    }

    return "rtspsrc location=\"" + config.rtsp_url + "\" latency=" + std::to_string(config.rtsp_latency_ms) +
           " drop-on-latency=true protocols=" + config.rtsp_transport +
           " ! " + depay +
           " ! " + parse +
           " ! mppvideodec" + decode_output_queue() +
           bgr_convert(config) +
           " ! appsink drop=true max-buffers=1 sync=false";
}

static bool write_clip_mp4(const std::vector<RecordedFrame> &frames,
                           const std::string &output_path,
                           double fps)
{
    if (frames.empty()) {
        printf("%s[save] no processed frames buffered yet%s\n", LOG_YELLOW, LOG_RESET);
        return false;
    }

    cv::Size frame_size(frames[0].bgr_frame.cols, frames[0].bgr_frame.rows);
    if (frame_size.width <= 0 || frame_size.height <= 0) {
        printf("%s[save] buffered frame has invalid size%s\n", LOG_RED, LOG_RESET);
        return false;
    }

    std::string tmp_path = output_path + ".tmp";
    remove(tmp_path.c_str());

    double start_ms = get_current_time_ms();
    std::string out_pipeline = "appsrc ! videoconvert ! video/x-raw,format=NV12 ! mpph264enc ! h264parse ! mp4mux ! filesink location=" + tmp_path;
    cv::VideoWriter writer(out_pipeline, cv::CAP_GSTREAMER, 0, fps, frame_size);
    if (!writer.isOpened()) {
        printf("%s[save] failed to open writer pipeline for %s%s\n", LOG_RED, tmp_path.c_str(), LOG_RESET);
        return false;
    }

    for (const RecordedFrame &frame : frames) {
        writer.write(frame.bgr_frame);
    }
    writer.release();

    if (rename(tmp_path.c_str(), output_path.c_str()) != 0) {
        printf("%s[save] failed to rename %s to %s: %s%s\n",
               LOG_RED, tmp_path.c_str(), output_path.c_str(), strerror(errno), LOG_RESET);
        return false;
    }

    double save_ms = get_current_time_ms() - start_ms;
    printf("%s[save] wrote %zu frames to %s at %.2f fps in %.2f ms%s\n",
           LOG_CYAN, frames.size(), output_path.c_str(), fps, save_ms, LOG_RESET);
    return true;
}

static void input_worker(std::atomic<bool> *stop_requested,
                         std::atomic<int> *save_requests,
                         std::condition_variable *queue_cv,
                         const std::string output_path)
{
    TerminalModeGuard terminal_guard;
    if (!terminal_guard.enabled()) {
        printf("%s[input] stdin is not a TTY; keyboard trigger disabled%s\n", LOG_YELLOW, LOG_RESET);
        return;
    }

    printf("%s[input] press 's' to save %s, 'q' to quit%s\n",
           LOG_CYAN, output_path.c_str(), LOG_RESET);
    while (!stop_requested->load()) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 100000;

        int ready = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &timeout);
        if (ready <= 0) {
            continue;
        }

        char ch = 0;
        ssize_t bytes_read = read(STDIN_FILENO, &ch, 1);
        if (bytes_read <= 0) {
            continue;
        }

        if (ch == 's' || ch == 'S') {
            save_requests->fetch_add(1);
        } else if (ch == 'q' || ch == 'Q') {
            stop_requested->store(true);
            queue_cv->notify_all();
            break;
        }
    }
}

static void complete_frame(FramePacket *packet,
                           RecorderBuffer *recorder,
                           RuntimeStats *stats,
                           RollingTiming *processing_timing,
                           std::condition_variable *queue_cv,
                           RetinaOverlayCache *retina_overlay_cache,
                           const char *color,
                           int worker_id,
                           int ctx_id)
{
    if (packet->has_yolo) {
        draw_yolo_results(&packet->bgr_frame, packet->od_results);
    }
    if (retina_overlay_cache != NULL) {
        retina_overlay_cache->draw(&packet->bgr_frame);
    }

    double end_to_end_ms = 0.0;
    if (packet->processing_start_ms > 0.0) {
        end_to_end_ms = get_current_time_ms() - packet->processing_start_ms;
        processing_timing->add(end_to_end_ms);
        log_stage_time(color, "e2e", packet->seq, worker_id, ctx_id, end_to_end_ms);
    }

    recorder->add_completed(packet->seq, std::move(packet->bgr_frame));
    stats->processed_frames.fetch_add(1);
    queue_cv->notify_all();
}

static void yolo_worker(rknn_app_context_t *yolo_ctx,
                        std::deque<FramePacket> *input_queue,
                        std::deque<FramePacket> *retina_queue,
                        std::mutex *queue_mutex,
                        std::condition_variable *queue_cv,
                        std::atomic<bool> *capture_done,
                        std::atomic<bool> *stop_requested,
                        RecorderBuffer *recorder,
                        size_t max_retina_queue_size,
                        DetectionCache *detection_cache,
                        RetinaOverlayCache *retina_overlay_cache,
                        RuntimeStats *stats,
                        RollingTiming *yolo_prep_timing,
                        RollingTiming *yolo_npu_timing,
                        RollingTiming *processing_timing,
                        RollingTiming *yolo_rknn_perf_timing,
                        bool rknn_perf,
                        bool async_retina,
                        int retina_reinfer_interval,
                        int worker_id)
{
    while (true) {
        FramePacket packet;

        {
            std::unique_lock<std::mutex> lock(*queue_mutex);
            queue_cv->wait(lock, [&]() {
                return !input_queue->empty() || capture_done->load() || stop_requested->load();
            });

            if (stop_requested->load()) {
                return;
            }

            if (input_queue->empty()) {
                if (capture_done->load()) {
                    break;
                }
                continue;
            }

            packet = std::move(input_queue->front());
            input_queue->pop_front();
            queue_cv->notify_all();
        }

        double yolo_prep_start_ms = get_current_time_ms();
        cv::cvtColor(packet.bgr_frame, packet.rgb_frame, cv::COLOR_BGR2RGB);

        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.width = packet.rgb_frame.cols;
        src_image.height = packet.rgb_frame.rows;
        src_image.width_stride = packet.rgb_frame.cols;
        src_image.height_stride = packet.rgb_frame.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.virt_addr = packet.rgb_frame.data;
        src_image.size = packet.rgb_frame.total() * packet.rgb_frame.channels();
        double yolo_prep_ms = get_current_time_ms() - yolo_prep_start_ms;

        double yolo_npu_start_ms = get_current_time_ms();
        int ret = inference_yolo11_model(yolo_ctx, &src_image, &packet.od_results);
        double yolo_npu_ms = get_current_time_ms() - yolo_npu_start_ms;
        if (ret != 0) {
            printf("YOLO worker fail! ret=%d\n", ret);
            stop_requested->store(true);
            return;
        }

        packet.has_yolo = true;
        detection_cache->update(packet.od_results);
        stats->yolo_processed.fetch_add(1);
        yolo_prep_timing->add(yolo_prep_ms);
        yolo_npu_timing->add(yolo_npu_ms);
        log_stage_time(LOG_BLUE, "yolo-prep", packet.seq, worker_id, worker_id, yolo_prep_ms);
        log_stage_time(LOG_GREEN, "yolo-npu", packet.seq, worker_id, worker_id, yolo_npu_ms);

        if (rknn_perf && yolo_ctx->last_perf_run_us >= 0) {
            double perf_ms = yolo_ctx->last_perf_run_us / 1000.0;
            yolo_rknn_perf_timing->add(perf_ms);
            log_stage_time(LOG_GREEN, "yolo-rknn", packet.seq, worker_id, worker_id, perf_ms);
        }

        packet.run_retina = (retina_reinfer_interval <= 1 ||
                             (packet.seq % static_cast<size_t>(retina_reinfer_interval)) == 0);

        if (async_retina) {
            FramePacket output_packet;
            output_packet.seq = packet.seq;
            output_packet.bgr_frame = packet.bgr_frame.clone();
            output_packet.od_results = packet.od_results;
            output_packet.processing_start_ms = packet.processing_start_ms;
            output_packet.has_yolo = true;
            complete_frame(&output_packet,
                           recorder,
                           stats,
                           processing_timing,
                           queue_cv,
                           retina_overlay_cache,
                           LOG_YELLOW,
                           worker_id,
                           worker_id);
            stats->async_retina_outputs.fetch_add(1);
            packet.record_on_retina = false;
        }

        if (!packet.run_retina) {
            stats->retina_interval_skips.fetch_add(1);
            if (!async_retina) {
                complete_frame(&packet,
                               recorder,
                               stats,
                               processing_timing,
                               queue_cv,
                               retina_overlay_cache,
                               LOG_YELLOW,
                               worker_id,
                               worker_id);
            }
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(*queue_mutex);
            while (retina_queue->size() >= max_retina_queue_size) {
                FramePacket dropped = std::move(retina_queue->front());
                retina_queue->pop_front();
                if (dropped.record_on_retina) {
                    recorder->mark_dropped(dropped.seq);
                }
                stats->retina_queue_drops.fetch_add(1);
                printf("%s[drop][worker-%d][frame-%zu] retina backlog full, dropping oldest frame%s\n",
                       LOG_YELLOW, worker_id, dropped.seq, LOG_RESET);
            }
            retina_queue->push_back(std::move(packet));
        }
        queue_cv->notify_all();
    }
}

static void retina_worker(retina_app_context_t *retina_ctx,
                          std::deque<FramePacket> *retina_queue,
                          std::mutex *queue_mutex,
                          std::condition_variable *queue_cv,
                          std::atomic<bool> *capture_done,
                          std::atomic<bool> *stop_requested,
                          RecorderBuffer *recorder,
                          RetinaOverlayCache *retina_overlay_cache,
                          int min_retina_crop_size,
                          int max_retina_crops,
                          RuntimeStats *stats,
                          RollingTiming *processing_timing,
                          RollingTiming *retina_npu_timing,
                          RollingTiming *retina_draw_timing,
                          RollingTiming *retina_rknn_perf_timing,
                          bool rknn_perf,
                          int worker_id)
{
    while (true) {
        FramePacket packet;

        {
            std::unique_lock<std::mutex> lock(*queue_mutex);
            queue_cv->wait(lock, [&]() {
                return !retina_queue->empty() || capture_done->load() || stop_requested->load();
            });

            if (stop_requested->load()) {
                return;
            }

            if (retina_queue->empty()) {
                if (capture_done->load()) {
                    break;
                }
                continue;
            }

            packet = std::move(retina_queue->front());
            retina_queue->pop_front();
            queue_cv->notify_all();
        }

        double retina_npu_start_ms = 0.0;
        double retina_draw_start_ms = 0.0;
        if (!packet.run_retina) {
            stats->retina_interval_skips.fetch_add(1);
            if (packet.record_on_retina) {
                complete_frame(&packet,
                               recorder,
                               stats,
                               processing_timing,
                               queue_cv,
                               retina_overlay_cache,
                               LOG_YELLOW,
                               worker_id,
                               worker_id);
            }
            continue;
        }

        if (packet.has_yolo) {
            retina_npu_start_ms = get_current_time_ms();
            retina_draw_start_ms = 0.0;
            int retina_crops_run = 0;
            double retina_rknn_perf_ms = 0.0;
            std::vector<FaceLandmarks> frame_landmarks;
            for (int i = 0; i < packet.od_results.count; i++) {
                object_detect_result *det_result = &(packet.od_results.results[i]);

                if (det_result->cls_id == 0) {
                    int x1 = std::max(0, det_result->box.left) & (~15);
                    int y1 = std::max(0, det_result->box.top) & (~15);
                    int x2 = std::min(packet.rgb_frame.cols - 1, det_result->box.right);
                    int y2 = std::min(packet.rgb_frame.rows - 1, det_result->box.bottom);

                    int crop_w = (x2 - x1) & (~15);
                    int crop_h = (y2 - y1) & (~15);

                    if (crop_w <= 0 || crop_h <= 0 ||
                        crop_w < min_retina_crop_size || crop_h < min_retina_crop_size) {
                        stats->retina_crops_skipped_small.fetch_add(1);
                        continue;
                    }

                    if (retina_crops_run >= max_retina_crops) {
                        stats->retina_crops_skipped_limit.fetch_add(1);
                        continue;
                    }

                    {
                        retina_crops_run++;
                        stats->retina_crops_run.fetch_add(1);
                        cv::Mat person_crop = packet.rgb_frame(cv::Rect(x1, y1, crop_w, crop_h)).clone();

                        image_buffer_t crop_img;
                        memset(&crop_img, 0, sizeof(image_buffer_t));
                        crop_img.width = person_crop.cols;
                        crop_img.height = person_crop.rows;
                        crop_img.width_stride = person_crop.cols;
                        crop_img.height_stride = person_crop.rows;
                        crop_img.format = IMAGE_FORMAT_RGB888;
                        crop_img.virt_addr = person_crop.data;
                        crop_img.size = person_crop.total() * person_crop.channels();

                        retinaface_result retina_res;
                        int ret = inference_retinaface_model(retina_ctx, &crop_img, &retina_res);
                        if (ret != 0) {
                            printf("Retina worker fail! ret=%d\n", ret);
                            stop_requested->store(true);
                            return;
                        }

                        if (rknn_perf && retina_ctx->last_perf_run_us >= 0) {
                            retina_rknn_perf_ms += retina_ctx->last_perf_run_us / 1000.0;
                        }

                        for (int f = 0; f < retina_res.count; f++) {
                            if (retina_res.object[f].score > 0.5) {
                                FaceLandmarks landmarks;
                                for (int p = 0; p < 5; p++) {
                                    int global_x = x1 + retina_res.object[f].ponit[p].x;
                                    int global_y = y1 + retina_res.object[f].ponit[p].y;
                                    landmarks.points[p] = cv::Point(global_x, global_y);
                                    if (packet.record_on_retina) {
                                        cv::circle(packet.bgr_frame, landmarks.points[p], 4, cv::Scalar(0, 165, 255), -1);
                                    }
                                }
                                frame_landmarks.push_back(landmarks);
                            }
                        }
                    }
                }
            }
            double retina_npu_ms = get_current_time_ms() - retina_npu_start_ms;
            retina_npu_timing->add(retina_npu_ms);
            log_stage_time(LOG_CYAN, "retina-npu", packet.seq, worker_id, worker_id, retina_npu_ms);
            if (retina_overlay_cache != NULL) {
                retina_overlay_cache->update(std::move(frame_landmarks));
            }

            if (rknn_perf) {
                retina_rknn_perf_timing->add(retina_rknn_perf_ms);
                log_stage_time(LOG_CYAN, "retina-rknn", packet.seq, worker_id, worker_id, retina_rknn_perf_ms);
            }

            if (packet.record_on_retina) {
                retina_draw_start_ms = get_current_time_ms();
                draw_yolo_results(&packet.bgr_frame, packet.od_results);
            }
        }

        packet.has_retina = true;
        double retina_draw_ms = 0.0;
        if (retina_draw_start_ms > 0.0) {
            retina_draw_ms = get_current_time_ms() - retina_draw_start_ms;
            retina_draw_timing->add(retina_draw_ms);
            log_stage_time(LOG_BLUE, "retina-draw", packet.seq, worker_id, worker_id, retina_draw_ms);
        }

        if (packet.record_on_retina) {
            double end_to_end_ms = 0.0;
            if (packet.processing_start_ms > 0.0) {
                end_to_end_ms = get_current_time_ms() - packet.processing_start_ms;
                processing_timing->add(end_to_end_ms);
                log_stage_time(LOG_YELLOW, "e2e", packet.seq, worker_id, worker_id, end_to_end_ms);
            }

            recorder->add_completed(packet.seq, std::move(packet.bgr_frame));
            stats->processed_frames.fetch_add(1);
        }
        queue_cv->notify_all();
    }
}

static rknn_core_mask round_robin_npu_core_mask(bool is_yolo,
                                                int worker_id,
                                                int yolo_worker_count)
{
    const rknn_core_mask single_core_masks[3] = {
        RKNN_NPU_CORE_0,
        RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_2,
    };
    int slot = is_yolo ? worker_id : yolo_worker_count + worker_id;
    return single_core_masks[slot % 3];
}

static rknn_core_mask npu_core_mask_for_worker(bool is_yolo,
                                               int worker_id,
                                               const AppConfig &config)
{
    if (config.npu_core_mode == "yolo-all") {
        return is_yolo ? RKNN_NPU_CORE_0_1_2 : RKNN_NPU_CORE_2;
    }

    if (config.npu_core_mode == "round-robin") {
        return round_robin_npu_core_mask(is_yolo, worker_id, config.yolo_worker_count);
    }

    if (config.yolo_worker_count == 1 && config.retina_worker_count == 1) {
        return is_yolo ? RKNN_NPU_CORE_0_1 : RKNN_NPU_CORE_2;
    }

    return round_robin_npu_core_mask(is_yolo, worker_id, config.yolo_worker_count);
}

static int npu_core_count(rknn_core_mask mask)
{
    int count = 0;
    if (mask & RKNN_NPU_CORE_0) {
        count++;
    }
    if (mask & RKNN_NPU_CORE_1) {
        count++;
    }
    if (mask & RKNN_NPU_CORE_2) {
        count++;
    }
    return std::max(1, count);
}

static const char *npu_core_mask_name(rknn_core_mask mask)
{
    if (mask == RKNN_NPU_CORE_AUTO) {
        return "auto";
    }
    if (mask == RKNN_NPU_CORE_0) {
        return "0";
    }
    if (mask == RKNN_NPU_CORE_1) {
        return "1";
    }
    if (mask == RKNN_NPU_CORE_2) {
        return "2";
    }
    if (mask == RKNN_NPU_CORE_0_1) {
        return "0_1";
    }
    if (mask == RKNN_NPU_CORE_0_1_2) {
        return "0_1_2";
    }
    if (mask == RKNN_NPU_CORE_ALL) {
        return "all";
    }
    return "unknown";
}

static void log_rknn_core_setup(const char *model_name,
                                int worker_id,
                                rknn_core_mask core_mask,
                                int set_core_ret,
                                int set_batch_ret)
{
    const char *color = (set_core_ret == RKNN_SUCC && set_batch_ret == RKNN_SUCC) ? LOG_CYAN : LOG_RED;
    printf("%s[npu-core] model=%s worker=%d requested=%s core_count=%d set_core_ret=%d set_batch_ret=%d%s\n",
           color,
           model_name,
           worker_id,
           npu_core_mask_name(core_mask),
           npu_core_count(core_mask),
           set_core_ret,
           set_batch_ret,
           LOG_RESET);
}

static void print_rolling_summary(size_t captured_frames,
                                  double run_start_ms,
                                  const RuntimeStats &stats,
                                  RollingTiming *capture_wait_timing,
                                  RollingTiming *capture_interval_timing,
                                  RollingTiming *yolo_prep_timing,
                                  RollingTiming *yolo_npu_timing,
                                  RollingTiming *retina_npu_timing,
                                  RollingTiming *retina_draw_timing,
                                  RollingTiming *processing_timing,
                                  RollingTiming *yolo_rknn_perf_timing,
                                  RollingTiming *retina_rknn_perf_timing,
                                  bool rknn_perf)
{
    double elapsed_ms = get_current_time_ms() - run_start_ms;
    double elapsed_s = elapsed_ms / 1000.0;
    if (elapsed_s <= 0.0) {
        elapsed_s = 1e-6;
    }

    TimingSnapshot capture_wait = capture_wait_timing->snapshot();
    TimingSnapshot capture_interval = capture_interval_timing->snapshot();
    TimingSnapshot yolo_prep = yolo_prep_timing->snapshot();
    TimingSnapshot yolo_npu = yolo_npu_timing->snapshot();
    TimingSnapshot retina_npu = retina_npu_timing->snapshot();
    TimingSnapshot retina_draw = retina_draw_timing->snapshot();
    TimingSnapshot processing = processing_timing->snapshot();

    printf("%s[rolling] captured=%zu captured_fps=%.2f processed=%zu processed_fps=%.2f yolo=%zu yolo_drops=%zu retina_drops=%zu interval_reuse=%zu retina_interval_skips=%zu async_outputs=%zu capture_wait(avg/p50/p95)=%.2f/%.2f/%.2f ms capture_interval(avg/p50/p95)=%.2f/%.2f/%.2f ms yolo_prep(avg/p50/p95)=%.2f/%.2f/%.2f ms yolo_npu(avg/p50/p95)=%.2f/%.2f/%.2f ms retina_npu(avg/p50/p95)=%.2f/%.2f/%.2f ms retina_draw(avg/p50/p95)=%.2f/%.2f/%.2f ms e2e(avg/p50/p95)=%.2f/%.2f/%.2f ms",
           LOG_CYAN,
           captured_frames,
           captured_frames / elapsed_s,
           stats.processed_frames.load(),
           stats.processed_frames.load() / elapsed_s,
           stats.yolo_processed.load(),
           stats.yolo_queue_drops.load(),
           stats.retina_queue_drops.load(),
           stats.interval_reuse_frames.load(),
           stats.retina_interval_skips.load(),
           stats.async_retina_outputs.load(),
           capture_wait.avg_ms,
           capture_wait.p50_ms,
           capture_wait.p95_ms,
           capture_interval.avg_ms,
           capture_interval.p50_ms,
           capture_interval.p95_ms,
           yolo_prep.avg_ms,
           yolo_prep.p50_ms,
           yolo_prep.p95_ms,
           yolo_npu.avg_ms,
           yolo_npu.p50_ms,
           yolo_npu.p95_ms,
           retina_npu.avg_ms,
           retina_npu.p50_ms,
           retina_npu.p95_ms,
           retina_draw.avg_ms,
           retina_draw.p50_ms,
           retina_draw.p95_ms,
           processing.avg_ms,
           processing.p50_ms,
           processing.p95_ms);
    if (rknn_perf) {
        TimingSnapshot yolo_rknn = yolo_rknn_perf_timing->snapshot();
        TimingSnapshot retina_rknn = retina_rknn_perf_timing->snapshot();
        printf(" yolo_rknn(avg/p50/p95)=%.2f/%.2f/%.2f ms retina_rknn(avg/p50/p95)=%.2f/%.2f/%.2f ms",
               yolo_rknn.avg_ms,
               yolo_rknn.p50_ms,
               yolo_rknn.p95_ms,
               retina_rknn.avg_ms,
               retina_rknn.p50_ms,
               retina_rknn.p95_ms);
    }
    printf("%s\n", LOG_RESET);
}

int main(int argc, char **argv)
{
    AppConfig config;
    if (!parse_config(argc, argv, &config)) {
        print_usage(argv[0]);
        return -1;
    }

    const char *yolo_path = argv[1];
    const char *retina_path = argv[2];
    int ret;

    init_post_process();

    std::string pipeline = build_input_pipeline(config);

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        printf("Failed to open live %s hardware video pipeline!\n", config.source.c_str());
        return -1;
    }

    std::deque<FramePacket> yolo_queue;
    std::deque<FramePacket> retina_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> capture_done(false);
    std::atomic<bool> stop_requested(false);
    std::atomic<int> save_requests(0);
    RuntimeStats stats;
    DetectionCache detection_cache;
    RetinaOverlayCache retina_overlay_cache;
    RollingTiming capture_wait_timing;
    RollingTiming capture_interval_timing;
    RollingTiming yolo_prep_timing;
    RollingTiming yolo_npu_timing;
    RollingTiming retina_npu_timing;
    RollingTiming retina_draw_timing;
    RollingTiming processing_timing;
    RollingTiming yolo_rknn_perf_timing;
    RollingTiming retina_rknn_perf_timing;
    const size_t max_reorder_pending = config.max_retina_queue_size + config.retina_worker_count + 8;
    RecorderBuffer recorder(config.max_buffered_frames, max_reorder_pending);
    uint32_t rknn_init_flags = RKNN_FLAG_ENABLE_SRAM;
    if (config.rknn_perf) {
        rknn_init_flags |= RKNN_FLAG_COLLECT_PERF_MASK;
    }

    std::vector<rknn_app_context_t> yolo_contexts(config.yolo_worker_count);
    std::vector<retina_app_context_t> retina_contexts(config.retina_worker_count);
    for (int i = 0; i < config.yolo_worker_count; ++i) {
        memset(&yolo_contexts[i], 0, sizeof(rknn_app_context_t));
        ret = init_yolo11_model(yolo_path, &yolo_contexts[i], rknn_init_flags);
        if (ret != 0) {
            for (int j = 0; j < i; ++j) {
                release_yolo11_model(&yolo_contexts[j]);
            }
            return -1;
        }
        yolo_contexts[i].collect_perf = config.rknn_perf;
        rknn_core_mask core_mask = npu_core_mask_for_worker(true, i, config);
        int set_core_ret = rknn_set_core_mask(yolo_contexts[i].rknn_ctx, core_mask);
        int set_batch_ret = rknn_set_batch_core_num(yolo_contexts[i].rknn_ctx, npu_core_count(core_mask));
        log_rknn_core_setup("yolo", i, core_mask, set_core_ret, set_batch_ret);
    }
    for (int i = 0; i < config.retina_worker_count; ++i) {
        memset(&retina_contexts[i], 0, sizeof(retina_app_context_t));
        ret = init_retinaface_model(retina_path, &retina_contexts[i], rknn_init_flags);
        if (ret != 0) {
            for (int j = 0; j < config.yolo_worker_count; ++j) {
                release_yolo11_model(&yolo_contexts[j]);
            }
            for (int j = 0; j < i; ++j) {
                release_retinaface_model(&retina_contexts[j]);
            }
            return -1;
        }
        retina_contexts[i].collect_perf = config.rknn_perf;
        rknn_core_mask core_mask = npu_core_mask_for_worker(false, i, config);
        int set_core_ret = rknn_set_core_mask(retina_contexts[i].rknn_ctx, core_mask);
        int set_batch_ret = rknn_set_batch_core_num(retina_contexts[i].rknn_ctx, npu_core_count(core_mask));
        log_rknn_core_setup("retina", i, core_mask, set_core_ret, set_batch_ret);
    }

    std::vector<std::thread> yolo_threads;
    std::vector<std::thread> retina_threads;
    yolo_threads.reserve(config.yolo_worker_count);
    retina_threads.reserve(config.retina_worker_count);
    for (int i = 0; i < config.yolo_worker_count; ++i) {
        yolo_threads.emplace_back(yolo_worker,
                                  &yolo_contexts[i],
                                  &yolo_queue,
                                  &retina_queue,
                                  &queue_mutex,
                                  &queue_cv,
                                  &capture_done,
                                  &stop_requested,
                                  &recorder,
                                  config.max_retina_queue_size,
                                  &detection_cache,
                                  &retina_overlay_cache,
                                  &stats,
                                  &yolo_prep_timing,
                                  &yolo_npu_timing,
                                  &processing_timing,
                                  &yolo_rknn_perf_timing,
                                  config.rknn_perf,
                                  config.async_retina,
                                  config.retina_reinfer_interval,
                                  i);
    }
    for (int i = 0; i < config.retina_worker_count; ++i) {
        retina_threads.emplace_back(retina_worker,
                                    &retina_contexts[i],
                                    &retina_queue,
                                    &queue_mutex,
                                    &queue_cv,
                                    &capture_done,
                                    &stop_requested,
                                    &recorder,
                                    &retina_overlay_cache,
                                    config.min_retina_crop_size,
                                    config.max_retina_crops,
                                    &stats,
                                    &processing_timing,
                                    &retina_npu_timing,
                                    &retina_draw_timing,
                                    &retina_rknn_perf_timing,
                                    config.rknn_perf,
                                    i);
    }

    std::thread input_thread(input_worker, &stop_requested, &save_requests, &queue_cv, config.output_path);

    cv::Mat bgr_frame;
    size_t frame_count = 0;

    printf("--- Starting Live %s Cascaded Inference Loop ---\n", config.source.c_str());
    printf("source=%s codec=%s rtsp_transport=%s rtsp_latency_ms=%d udp_port=%d clip_frames=%zu max_buffered_frames=%zu fps=%.2f output=%s yolo_workers=%d retina_workers=%d max_yolo_queue=%zu max_retina_queue=%zu yolo_interval=%d retina_reinfer_interval=%d min_retina_crop=%d max_retina_crops=%d videoconvert_threads=%d npu_core_mode=%s rknn_perf=%d async_retina=%d\n",
           config.source.c_str(), config.camera_codec.c_str(), config.rtsp_transport.c_str(),
           config.rtsp_latency_ms, config.udp_port,
           config.clip_frames, config.max_buffered_frames,
           config.output_fps, config.output_path.c_str(),
           config.yolo_worker_count, config.retina_worker_count,
           config.max_yolo_queue_size, config.max_retina_queue_size,
           config.yolo_interval, config.retina_reinfer_interval,
           config.min_retina_crop_size,
           config.max_retina_crops, config.videoconvert_threads,
           config.npu_core_mode.c_str(), config.rknn_perf ? 1 : 0, config.async_retina ? 1 : 0);

    double run_start_ms = get_current_time_ms();
    double last_capture_success_ms = 0.0;
    double capture_wait_total_ms = 0.0;
    double capture_interval_total_ms = 0.0;
    double clone_total_ms = 0.0;
    while (true) {
        if (stop_requested.load()) {
            break;
        }

        int pending_saves = save_requests.exchange(0);
        for (int i = 0; i < pending_saves; ++i) {
            std::vector<RecordedFrame> frames = recorder.snapshot_last(config.clip_frames);
            write_clip_mp4(frames, config.output_path, config.output_fps);
        }

        double capture_wait_start_ms = get_current_time_ms();
        if (!cap.read(bgr_frame)) {
            printf("%s[capture] failed to read live frame; exiting%s\n", LOG_RED, LOG_RESET);
            break;
        }
        double capture_ready_ms = get_current_time_ms();
        double capture_wait_ms = capture_ready_ms - capture_wait_start_ms;
        capture_wait_total_ms += capture_wait_ms;
        capture_wait_timing.add(capture_wait_ms);
        log_stage_time(LOG_RED, "capture-wait", frame_count, 0, 0, capture_wait_ms);

        if (last_capture_success_ms > 0.0) {
            double capture_interval_ms = capture_ready_ms - last_capture_success_ms;
            capture_interval_total_ms += capture_interval_ms;
            capture_interval_timing.add(capture_interval_ms);
            log_stage_time(LOG_CYAN, "capture-interval", frame_count, 0, 0, capture_interval_ms);
        }
        last_capture_success_ms = capture_ready_ms;

        double clone_start_ms = get_current_time_ms();
        FramePacket packet;
        packet.seq = frame_count;
        packet.processing_start_ms = capture_ready_ms;
        packet.bgr_frame = bgr_frame.clone();
        double clone_ms = get_current_time_ms() - clone_start_ms;
        clone_total_ms += clone_ms;
        log_stage_time(LOG_BLUE, "capture-clone", packet.seq, 0, 0, clone_ms);

        if (config.yolo_interval > 1 && (frame_count % config.yolo_interval) != 0) {
            object_detect_result_list cached_results;
            if (detection_cache.snapshot(&cached_results)) {
                packet.od_results = cached_results;
                packet.has_yolo = true;
            }
            stats.interval_reuse_frames.fetch_add(1);
            complete_frame(&packet,
                           &recorder,
                           &stats,
                           &processing_timing,
                           &queue_cv,
                           &retina_overlay_cache,
                           LOG_YELLOW,
                           0,
                           0);
            frame_count++;
            if (frame_count % 100 == 0) {
                print_rolling_summary(frame_count,
                                      run_start_ms,
                                      stats,
                                      &capture_wait_timing,
                                      &capture_interval_timing,
                                      &yolo_prep_timing,
                                      &yolo_npu_timing,
                                      &retina_npu_timing,
                                      &retina_draw_timing,
                                      &processing_timing,
                                      &yolo_rknn_perf_timing,
                                      &retina_rknn_perf_timing,
                                      config.rknn_perf);
            }
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            while (yolo_queue.size() >= config.max_yolo_queue_size) {
                FramePacket dropped = std::move(yolo_queue.front());
                yolo_queue.pop_front();
                recorder.mark_dropped(dropped.seq);
                stats.yolo_queue_drops.fetch_add(1);
                printf("%s[drop][capture][frame-%zu] yolo backlog full, dropping oldest frame%s\n",
                       LOG_YELLOW, dropped.seq, LOG_RESET);
            }
            yolo_queue.push_back(std::move(packet));
        }
        queue_cv.notify_all();
        frame_count++;
        if (frame_count % 100 == 0) {
            print_rolling_summary(frame_count,
                                  run_start_ms,
                                  stats,
                                  &capture_wait_timing,
                                  &capture_interval_timing,
                                  &yolo_prep_timing,
                                  &yolo_npu_timing,
                                  &retina_npu_timing,
                                  &retina_draw_timing,
                                  &processing_timing,
                                  &yolo_rknn_perf_timing,
                                  &retina_rknn_perf_timing,
                                  config.rknn_perf);
        }
    }

    capture_done.store(true);
    stop_requested.store(true);
    queue_cv.notify_all();

    int pending_saves = save_requests.exchange(0);
    for (int i = 0; i < pending_saves; ++i) {
        std::vector<RecordedFrame> frames = recorder.snapshot_last(config.clip_frames);
        write_clip_mp4(frames, config.output_path, config.output_fps);
    }

    if (input_thread.joinable()) {
        input_thread.join();
    }
    for (auto &thread : yolo_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    for (auto &thread : retina_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    for (int i = 0; i < config.yolo_worker_count; ++i) {
        release_yolo11_model(&yolo_contexts[i]);
    }
    for (int i = 0; i < config.retina_worker_count; ++i) {
        release_retinaface_model(&retina_contexts[i]);
    }

    cap.release();
    deinit_post_process();
    TimingSnapshot capture_wait = capture_wait_timing.snapshot();
    TimingSnapshot capture_interval = capture_interval_timing.snapshot();
    TimingSnapshot yolo_prep = yolo_prep_timing.snapshot();
    TimingSnapshot yolo_npu = yolo_npu_timing.snapshot();
    TimingSnapshot retina_npu = retina_npu_timing.snapshot();
    TimingSnapshot retina_draw = retina_draw_timing.snapshot();
    TimingSnapshot processing = processing_timing.snapshot();
    printf("%s[summary] captured=%zu yolo_processed=%zu processed=%zu buffered=%zu yolo_drops=%zu retina_drops=%zu interval_reuse=%zu retina_interval_skips=%zu async_outputs=%zu retina_crops=%zu retina_small_skips=%zu retina_limit_skips=%zu capture_wait_total=%.2f ms capture_interval_total=%.2f ms clone=%.2f ms capture_wait(avg/p50/p95)=%.2f/%.2f/%.2f ms capture_interval(avg/p50/p95)=%.2f/%.2f/%.2f ms yolo_prep(avg/p50/p95)=%.2f/%.2f/%.2f ms yolo_npu(avg/p50/p95)=%.2f/%.2f/%.2f ms retina_npu(avg/p50/p95)=%.2f/%.2f/%.2f ms retina_draw(avg/p50/p95)=%.2f/%.2f/%.2f ms e2e(avg/p50/p95)=%.2f/%.2f/%.2f ms",
           LOG_CYAN,
           frame_count,
           stats.yolo_processed.load(),
           stats.processed_frames.load(),
           recorder.buffered_count(),
           stats.yolo_queue_drops.load(),
           stats.retina_queue_drops.load(),
           stats.interval_reuse_frames.load(),
           stats.retina_interval_skips.load(),
           stats.async_retina_outputs.load(),
           stats.retina_crops_run.load(),
           stats.retina_crops_skipped_small.load(),
           stats.retina_crops_skipped_limit.load(),
           capture_wait_total_ms,
           capture_interval_total_ms,
           clone_total_ms,
           capture_wait.avg_ms,
           capture_wait.p50_ms,
           capture_wait.p95_ms,
           capture_interval.avg_ms,
           capture_interval.p50_ms,
           capture_interval.p95_ms,
           yolo_prep.avg_ms,
           yolo_prep.p50_ms,
           yolo_prep.p95_ms,
           yolo_npu.avg_ms,
           yolo_npu.p50_ms,
           yolo_npu.p95_ms,
           retina_npu.avg_ms,
           retina_npu.p50_ms,
           retina_npu.p95_ms,
           retina_draw.avg_ms,
           retina_draw.p50_ms,
           retina_draw.p95_ms,
           processing.avg_ms,
           processing.p50_ms,
           processing.p95_ms);
    if (config.rknn_perf) {
        TimingSnapshot yolo_rknn = yolo_rknn_perf_timing.snapshot();
        TimingSnapshot retina_rknn = retina_rknn_perf_timing.snapshot();
        printf(" yolo_rknn(avg/p50/p95)=%.2f/%.2f/%.2f ms retina_rknn(avg/p50/p95)=%.2f/%.2f/%.2f ms",
               yolo_rknn.avg_ms,
               yolo_rknn.p50_ms,
               yolo_rknn.p95_ms,
               retina_rknn.avg_ms,
               retina_rknn.p50_ms,
               retina_rknn.p95_ms);
    }
    printf("%s\n", LOG_RESET);
    printf("Finished! Last saved clip path: '%s'\n", config.output_path.c_str());
    return 0;
}
