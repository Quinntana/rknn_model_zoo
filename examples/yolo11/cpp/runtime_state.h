// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#ifndef RKNN_YOLO11_RUNTIME_STATE_H_
#define RKNN_YOLO11_RUNTIME_STATE_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "app_config.h"
#include "yolo11.h"

extern const char *LOG_RESET;
extern const char *LOG_RED;
extern const char *LOG_GREEN;
extern const char *LOG_YELLOW;
extern const char *LOG_BLUE;
extern const char *LOG_CYAN;

double get_current_time_ms();
void log_stage_time(const char *color, const char *stage, size_t seq, int worker_id, int ctx_id, double ms);

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
    explicit RollingTiming(size_t max_samples = 256);

    void add(double ms);
    TimingSnapshot snapshot();

private:
    static double percentile(const std::vector<double> &sorted, double pct);

    std::mutex mutex_;
    std::deque<double> samples_;
    size_t max_samples_ = 0;
};

class DetectionCache {
public:
    void update(const object_detect_result_list &od_results);
    bool snapshot(object_detect_result_list *od_results);

private:
    std::mutex mutex_;
    object_detect_result_list od_results_;
    bool valid_ = false;
};

void draw_yolo_results(cv::Mat *bgr_frame, const object_detect_result_list &od_results);

class RecorderBuffer {
public:
    RecorderBuffer(size_t max_buffered_frames, size_t max_reorder_pending);

    void add_completed(size_t seq, cv::Mat bgr_frame);
    void mark_dropped(size_t seq);
    std::vector<RecordedFrame> snapshot_last(size_t frame_count);
    size_t buffered_count();

private:
    void flush_locked();

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
    void update(std::vector<FaceLandmarks> faces);
    void draw(cv::Mat *bgr_frame);

private:
    std::mutex mutex_;
    std::vector<FaceLandmarks> faces_;
};

rknn_core_mask npu_core_mask_for_worker(bool is_yolo, int worker_id, const AppConfig &config);
int npu_core_count(rknn_core_mask mask);
const char *npu_core_mask_name(rknn_core_mask mask);
void log_rknn_core_setup(const char *model_name, int worker_id, rknn_core_mask core_mask, int set_core_ret, int set_batch_ret);

void print_rolling_summary(size_t captured_frames,
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
                           bool rknn_perf);

void print_final_summary(size_t captured_frames,
                         const RuntimeStats &stats,
                         RecorderBuffer *recorder,
                         RollingTiming *capture_wait_timing,
                         RollingTiming *capture_interval_timing,
                         RollingTiming *yolo_prep_timing,
                         RollingTiming *yolo_npu_timing,
                         RollingTiming *retina_npu_timing,
                         RollingTiming *retina_draw_timing,
                         RollingTiming *processing_timing,
                         RollingTiming *yolo_rknn_perf_timing,
                         RollingTiming *retina_rknn_perf_timing,
                         bool rknn_perf,
                         double capture_wait_total_ms,
                         double capture_interval_total_ms,
                         double clone_total_ms);

#endif  // RKNN_YOLO11_RUNTIME_STATE_H_
