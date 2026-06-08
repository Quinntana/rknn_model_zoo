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
    int udp_port = 5000;
    size_t clip_frames = 150;
    size_t max_buffered_frames = 300;
    double output_fps = 10.0;
    std::string output_path = "output_cascaded.mp4";
};

struct FramePacket {
    size_t seq = 0;
    cv::Mat bgr_frame;
    cv::Mat rgb_frame;
    object_detect_result_list od_results;
    double source_read_start_ms = 0.0;
    bool has_yolo = false;
    bool has_retina = false;
    bool dropped = false;
};

struct RecordedFrame {
    size_t seq = 0;
    cv::Mat bgr_frame;
};

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
    printf("Usage: %s <yolo_model> <retina_model> [--udp-port 5000] [--clip-frames 150] [--max-buffered-frames 300] [--fps 10] [--output output_cascaded.mp4]\n", program);
}

static bool parse_config(int argc, char **argv, AppConfig *config)
{
    if (argc < 3) {
        return false;
    }

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--udp-port") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->udp_port)) {
                return false;
            }
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
        } else {
            return false;
        }
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
    return true;
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

static void yolo_worker(rknn_app_context_t *yolo_ctx,
                        std::deque<FramePacket> *input_queue,
                        std::deque<FramePacket> *retina_queue,
                        std::mutex *queue_mutex,
                        std::condition_variable *queue_cv,
                        std::atomic<bool> *capture_done,
                        std::atomic<bool> *stop_requested,
                        RecorderBuffer *recorder,
                        size_t max_retina_queue_size,
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
        log_stage_time(LOG_BLUE, "yolo-prep", packet.seq, worker_id, worker_id, yolo_prep_ms);
        log_stage_time(LOG_GREEN, "yolo-npu", packet.seq, worker_id, worker_id, yolo_npu_ms);

        {
            std::unique_lock<std::mutex> lock(*queue_mutex);
            if (retina_queue->size() >= max_retina_queue_size) {
                FramePacket dropped = std::move(retina_queue->front());
                retina_queue->pop_front();
                recorder->mark_dropped(dropped.seq);
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
        if (packet.has_yolo) {
            retina_npu_start_ms = get_current_time_ms();
            retina_draw_start_ms = 0.0;
            for (int i = 0; i < packet.od_results.count; i++) {
                object_detect_result *det_result = &(packet.od_results.results[i]);

                if (det_result->cls_id == 0) {
                    int x1 = std::max(0, det_result->box.left) & (~15);
                    int y1 = std::max(0, det_result->box.top) & (~15);
                    int x2 = std::min(packet.rgb_frame.cols - 1, det_result->box.right);
                    int y2 = std::min(packet.rgb_frame.rows - 1, det_result->box.bottom);

                    int crop_w = (x2 - x1) & (~15);
                    int crop_h = (y2 - y1) & (~15);

                    if (crop_w > 0 && crop_h > 0) {
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

                        for (int f = 0; f < retina_res.count; f++) {
                            if (retina_res.object[f].score > 0.5) {
                                for (int p = 0; p < 5; p++) {
                                    int global_x = x1 + retina_res.object[f].ponit[p].x;
                                    int global_y = y1 + retina_res.object[f].ponit[p].y;
                                    cv::circle(packet.bgr_frame, cv::Point(global_x, global_y), 4, cv::Scalar(0, 165, 255), -1);
                                }
                            }
                        }
                    }
                }
            }
            double retina_npu_ms = get_current_time_ms() - retina_npu_start_ms;
            log_stage_time(LOG_CYAN, "retina-npu", packet.seq, worker_id, worker_id, retina_npu_ms);

            retina_draw_start_ms = get_current_time_ms();
            for (int i = 0; i < packet.od_results.count; i++) {
                object_detect_result *det_result = &(packet.od_results.results[i]);

                cv::rectangle(packet.bgr_frame,
                              cv::Point(det_result->box.left, det_result->box.top),
                              cv::Point(det_result->box.right, det_result->box.bottom),
                              cv::Scalar(0, 255, 0), 2);
                char text[256];
                sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
                cv::putText(packet.bgr_frame, text,
                            cv::Point(det_result->box.left, det_result->box.top - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
            }
        }

        packet.has_retina = true;
        double retina_draw_ms = 0.0;
        if (retina_draw_start_ms > 0.0) {
            retina_draw_ms = get_current_time_ms() - retina_draw_start_ms;
            log_stage_time(LOG_BLUE, "retina-draw", packet.seq, worker_id, worker_id, retina_draw_ms);
        }

        double end_to_end_ms = 0.0;
        if (packet.source_read_start_ms > 0.0) {
            end_to_end_ms = get_current_time_ms() - packet.source_read_start_ms;
            log_stage_time(LOG_YELLOW, "e2e", packet.seq, worker_id, worker_id, end_to_end_ms);
        }

        {
            recorder->add_completed(packet.seq, std::move(packet.bgr_frame));
        }
        queue_cv->notify_all();
    }
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

    std::string pipeline = "udpsrc port=" + std::to_string(config.udp_port) +
                           " caps=\"video/mpegts, systemstream=(boolean)true, packetsize=(int)188\""
                           " ! tsdemux ! h264parse ! mppvideodec ! videoconvert ! video/x-raw,format=BGR"
                           " ! appsink drop=true max-buffers=1 sync=false";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        printf("Failed to open live UDP hardware video pipeline!\n");
        return -1;
    }

    std::deque<FramePacket> yolo_queue;
    std::deque<FramePacket> retina_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> capture_done(false);
    std::atomic<bool> stop_requested(false);
    std::atomic<int> save_requests(0);
    const size_t max_yolo_queue_size = 4;
    const size_t max_retina_queue_size = 8;
    const int yolo_worker_count = 3;
    const int retina_worker_count = 3;
    const size_t max_reorder_pending = max_retina_queue_size + retina_worker_count + 8;
    RecorderBuffer recorder(config.max_buffered_frames, max_reorder_pending);

    std::vector<rknn_app_context_t> yolo_contexts(yolo_worker_count);
    std::vector<retina_app_context_t> retina_contexts(retina_worker_count);
    for (int i = 0; i < yolo_worker_count; ++i) {
        memset(&yolo_contexts[i], 0, sizeof(rknn_app_context_t));
        ret = init_yolo11_model(yolo_path, &yolo_contexts[i]);
        if (ret != 0) {
            for (int j = 0; j < i; ++j) {
                release_yolo11_model(&yolo_contexts[j]);
            }
            return -1;
        }
        rknn_set_core_mask(yolo_contexts[i].rknn_ctx, RKNN_NPU_CORE_0_1_2);
        ret = rknn_set_batch_core_num(yolo_contexts[i].rknn_ctx, 3);
        if (ret != RKNN_SUCC) {
            printf("rknn_set_batch_core_num(yolo-%d) fail! ret=%d\n", i, ret);
        }
    }
    for (int i = 0; i < retina_worker_count; ++i) {
        memset(&retina_contexts[i], 0, sizeof(retina_app_context_t));
        ret = init_retinaface_model(retina_path, &retina_contexts[i]);
        if (ret != 0) {
            for (int j = 0; j < yolo_worker_count; ++j) {
                release_yolo11_model(&yolo_contexts[j]);
            }
            for (int j = 0; j < i; ++j) {
                release_retinaface_model(&retina_contexts[j]);
            }
            return -1;
        }
        rknn_set_core_mask(retina_contexts[i].rknn_ctx, RKNN_NPU_CORE_0_1_2);
        ret = rknn_set_batch_core_num(retina_contexts[i].rknn_ctx, 3);
        if (ret != RKNN_SUCC) {
            printf("rknn_set_batch_core_num(retina-%d) fail! ret=%d\n", i, ret);
        }
    }

    std::vector<std::thread> yolo_threads;
    std::vector<std::thread> retina_threads;
    yolo_threads.reserve(yolo_worker_count);
    retina_threads.reserve(retina_worker_count);
    for (int i = 0; i < yolo_worker_count; ++i) {
        yolo_threads.emplace_back(yolo_worker,
                                  &yolo_contexts[i],
                                  &yolo_queue,
                                  &retina_queue,
                                  &queue_mutex,
                                  &queue_cv,
                                  &capture_done,
                                  &stop_requested,
                                  &recorder,
                                  max_retina_queue_size,
                                  i);
    }
    for (int i = 0; i < retina_worker_count; ++i) {
        retina_threads.emplace_back(retina_worker,
                                    &retina_contexts[i],
                                    &retina_queue,
                                    &queue_mutex,
                                    &queue_cv,
                                    &capture_done,
                                    &stop_requested,
                                    &recorder,
                                    i);
    }

    std::thread input_thread(input_worker, &stop_requested, &save_requests, &queue_cv, config.output_path);

    cv::Mat bgr_frame;
    size_t frame_count = 0;

    printf("--- Starting Live UDP Cascaded Inference Loop ---\n");
    printf("UDP port=%d clip_frames=%zu max_buffered_frames=%zu fps=%.2f output=%s\n",
           config.udp_port, config.clip_frames, config.max_buffered_frames,
           config.output_fps, config.output_path.c_str());

    double source_read_total_ms = 0.0;
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

        double source_read_start_ms = get_current_time_ms();
        if (!cap.read(bgr_frame)) {
            printf("%s[capture] failed to read live frame; exiting%s\n", LOG_RED, LOG_RESET);
            break;
        }
        double source_read_ms = get_current_time_ms() - source_read_start_ms;
        source_read_total_ms += source_read_ms;
        log_stage_time(LOG_RED, "source-read", frame_count, 0, 0, source_read_ms);

        double clone_start_ms = get_current_time_ms();
        FramePacket packet;
        packet.seq = frame_count;
        packet.source_read_start_ms = source_read_start_ms;
        packet.bgr_frame = bgr_frame.clone();
        double clone_ms = get_current_time_ms() - clone_start_ms;
        clone_total_ms += clone_ms;
        log_stage_time(LOG_BLUE, "capture-clone", packet.seq, 0, 0, clone_ms);

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (yolo_queue.size() >= max_yolo_queue_size) {
                FramePacket dropped = std::move(yolo_queue.front());
                yolo_queue.pop_front();
                recorder.mark_dropped(dropped.seq);
                printf("%s[drop][capture][frame-%zu] yolo backlog full, dropping oldest frame%s\n",
                       LOG_YELLOW, dropped.seq, LOG_RESET);
            }
            yolo_queue.push_back(std::move(packet));
        }
        queue_cv.notify_all();
        frame_count++;
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

    for (int i = 0; i < yolo_worker_count; ++i) {
        release_yolo11_model(&yolo_contexts[i]);
    }
    for (int i = 0; i < retina_worker_count; ++i) {
        release_retinaface_model(&retina_contexts[i]);
    }

    cap.release();
    deinit_post_process();
    printf("%s[summary] captured=%zu buffered=%zu source_read=%.2f ms clone=%.2f ms%s\n",
           LOG_CYAN, frame_count, recorder.buffered_count(), source_read_total_ms, clone_total_ms, LOG_RESET);
    printf("Finished! Last saved clip path: '%s'\n", config.output_path.c_str());
    return 0;
}
