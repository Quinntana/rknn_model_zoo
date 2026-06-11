// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#ifndef RKNN_YOLO11_WORKER_THREADS_H_
#define RKNN_YOLO11_WORKER_THREADS_H_

#include <stddef.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "retinaface.h"
#include "runtime_state.h"
#include "yolo11.h"

class LiveStreamer;

bool write_clip_mp4(const std::vector<RecordedFrame> &frames, const std::string &output_path, double fps);

void input_worker(std::atomic<bool> *stop_requested,
                  std::atomic<int> *save_requests,
                  std::condition_variable *queue_cv,
                  const std::string output_path);

void complete_frame(FramePacket *packet,
                    RecorderBuffer *recorder,
                    LiveStreamer *live_streamer,
                    RuntimeStats *stats,
                    RollingTiming *processing_timing,
                    std::condition_variable *queue_cv,
                    RetinaOverlayCache *retina_overlay_cache,
                    const char *color,
                    int worker_id,
                    int ctx_id);

void yolo_worker(rknn_app_context_t *yolo_ctx,
                 std::deque<FramePacket> *input_queue,
                 std::deque<FramePacket> *retina_queue,
                 std::mutex *queue_mutex,
                 std::condition_variable *queue_cv,
                 std::atomic<bool> *capture_done,
                 std::atomic<bool> *stop_requested,
                 RecorderBuffer *recorder,
                 LiveStreamer *live_streamer,
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
                 int worker_id);

void retina_worker(retina_app_context_t *retina_ctx,
                   std::deque<FramePacket> *retina_queue,
                   std::mutex *queue_mutex,
                   std::condition_variable *queue_cv,
                   std::atomic<bool> *capture_done,
                   std::atomic<bool> *stop_requested,
                   RecorderBuffer *recorder,
                   LiveStreamer *live_streamer,
                   RetinaOverlayCache *retina_overlay_cache,
                   int min_retina_crop_size,
                   int max_retina_crops,
                   RuntimeStats *stats,
                   RollingTiming *processing_timing,
                   RollingTiming *retina_npu_timing,
                   RollingTiming *retina_draw_timing,
                   RollingTiming *retina_rknn_perf_timing,
                   bool rknn_perf,
                   int worker_id);

#endif  // RKNN_YOLO11_WORKER_THREADS_H_
