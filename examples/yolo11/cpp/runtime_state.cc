// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include "runtime_state.h"

#include <stdio.h>
#include <sys/time.h>

#include <algorithm>

const char *LOG_RESET = "\033[0m";
const char *LOG_RED = "\033[31m";
const char *LOG_GREEN = "\033[32m";
const char *LOG_YELLOW = "\033[33m";
const char *LOG_BLUE = "\033[34m";
const char *LOG_CYAN = "\033[36m";

double get_current_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void log_stage_time(const char *color, const char *stage, size_t seq, int worker_id, int ctx_id, double ms)
{
    printf("%s[%s][worker-%d][ctx-%d][frame-%zu] %.2f ms%s\n", color, stage, worker_id, ctx_id, seq, ms, LOG_RESET);
}

RollingTiming::RollingTiming(size_t max_samples)
    : max_samples_(max_samples)
{
}

void RollingTiming::add(double ms)
{
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.push_back(ms);
    while (samples_.size() > max_samples_) {
        samples_.pop_front();
    }
}

TimingSnapshot RollingTiming::snapshot()
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

double RollingTiming::percentile(const std::vector<double> &sorted, double pct)
{
    if (sorted.empty()) {
        return 0.0;
    }
    size_t index = (size_t)((sorted.size() - 1) * pct + 0.5);
    index = std::min(index, sorted.size() - 1);
    return sorted[index];
}

void DetectionCache::update(const object_detect_result_list &od_results)
{
    std::lock_guard<std::mutex> lock(mutex_);
    od_results_ = od_results;
    valid_ = true;
}

bool DetectionCache::snapshot(object_detect_result_list *od_results)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_) {
        return false;
    }
    *od_results = od_results_;
    return true;
}

void draw_yolo_results(cv::Mat *bgr_frame, const object_detect_result_list &od_results)
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

RecorderBuffer::RecorderBuffer(size_t max_buffered_frames, size_t max_reorder_pending)
    : max_buffered_frames_(max_buffered_frames),
      max_reorder_pending_(max_reorder_pending)
{
}

void RecorderBuffer::add_completed(size_t seq, cv::Mat bgr_frame)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (seq < next_seq_) {
        return;
    }
    pending_[seq] = RecordedFrame{seq, std::move(bgr_frame)};
    flush_locked();
}

void RecorderBuffer::mark_dropped(size_t seq)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (seq < next_seq_) {
        return;
    }
    dropped_.insert(seq);
    flush_locked();
}

std::vector<RecordedFrame> RecorderBuffer::snapshot_last(size_t frame_count)
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

size_t RecorderBuffer::buffered_count()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return ring_.size();
}

void RecorderBuffer::flush_locked()
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

void RetinaOverlayCache::update(std::vector<FaceLandmarks> faces)
{
    std::lock_guard<std::mutex> lock(mutex_);
    faces_ = std::move(faces);
}

void RetinaOverlayCache::draw(cv::Mat *bgr_frame)
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

static rknn_core_mask round_robin_npu_core_mask(bool is_yolo, int worker_id, int yolo_worker_count)
{
    const rknn_core_mask single_core_masks[3] = {
        RKNN_NPU_CORE_0,
        RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_2,
    };
    int slot = is_yolo ? worker_id : yolo_worker_count + worker_id;
    return single_core_masks[slot % 3];
}

rknn_core_mask npu_core_mask_for_worker(bool is_yolo, int worker_id, const AppConfig &config)
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

int npu_core_count(rknn_core_mask mask)
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

const char *npu_core_mask_name(rknn_core_mask mask)
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

void log_rknn_core_setup(const char *model_name,
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
                         double clone_total_ms)
{
    TimingSnapshot capture_wait = capture_wait_timing->snapshot();
    TimingSnapshot capture_interval = capture_interval_timing->snapshot();
    TimingSnapshot yolo_prep = yolo_prep_timing->snapshot();
    TimingSnapshot yolo_npu = yolo_npu_timing->snapshot();
    TimingSnapshot retina_npu = retina_npu_timing->snapshot();
    TimingSnapshot retina_draw = retina_draw_timing->snapshot();
    TimingSnapshot processing = processing_timing->snapshot();
    printf("%s[summary] captured=%zu yolo_processed=%zu processed=%zu buffered=%zu yolo_drops=%zu retina_drops=%zu interval_reuse=%zu retina_interval_skips=%zu async_outputs=%zu retina_crops=%zu retina_small_skips=%zu retina_limit_skips=%zu capture_wait_total=%.2f ms capture_interval_total=%.2f ms clone=%.2f ms capture_wait(avg/p50/p95)=%.2f/%.2f/%.2f ms capture_interval(avg/p50/p95)=%.2f/%.2f/%.2f ms yolo_prep(avg/p50/p95)=%.2f/%.2f/%.2f ms yolo_npu(avg/p50/p95)=%.2f/%.2f/%.2f ms retina_npu(avg/p50/p95)=%.2f/%.2f/%.2f ms retina_draw(avg/p50/p95)=%.2f/%.2f/%.2f ms e2e(avg/p50/p95)=%.2f/%.2f/%.2f ms",
           LOG_CYAN,
           captured_frames,
           stats.yolo_processed.load(),
           stats.processed_frames.load(),
           recorder->buffered_count(),
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
