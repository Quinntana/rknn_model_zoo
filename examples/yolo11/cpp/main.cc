// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "app_config.h"
#include "live_streamer.h"
#include "retinaface.h"
#include "runtime_state.h"
#include "worker_threads.h"
#include "yolo11.h"

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
    LiveStreamer live_streamer(config);
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
                                  &live_streamer,
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
                                    &live_streamer,
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
    printf("source=%s codec=%s rtsp_transport=%s rtsp_latency_ms=%d udp_port=%d clip_frames=%zu max_buffered_frames=%zu fps=%.2f output=%s yolo_workers=%d retina_workers=%d max_yolo_queue=%zu max_retina_queue=%zu yolo_interval=%d retina_reinfer_interval=%d min_retina_crop=%d max_retina_crops=%d videoconvert_threads=%d npu_core_mode=%s rknn_perf=%d async_retina=%d serve_rtsp=%d stream_port=%d stream_path=%s stream_fps=%.2f stream_bitrate_kbps=%d stream_encoder=%s\n",
           config.source.c_str(), config.camera_codec.c_str(), config.rtsp_transport.c_str(),
           config.rtsp_latency_ms, config.udp_port,
           config.clip_frames, config.max_buffered_frames,
           config.output_fps, config.output_path.c_str(),
           config.yolo_worker_count, config.retina_worker_count,
           config.max_yolo_queue_size, config.max_retina_queue_size,
           config.yolo_interval, config.retina_reinfer_interval,
           config.min_retina_crop_size,
           config.max_retina_crops, config.videoconvert_threads,
           config.npu_core_mode.c_str(), config.rknn_perf ? 1 : 0, config.async_retina ? 1 : 0,
           config.stream_enabled ? 1 : 0, config.stream_port, config.stream_path.c_str(),
           config.stream_fps, config.stream_bitrate_kbps, config.stream_encoder.c_str());

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
                           &live_streamer,
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
    live_streamer.stop();

    for (int i = 0; i < config.yolo_worker_count; ++i) {
        release_yolo11_model(&yolo_contexts[i]);
    }
    for (int i = 0; i < config.retina_worker_count; ++i) {
        release_retinaface_model(&retina_contexts[i]);
    }

    cap.release();
    deinit_post_process();
    print_final_summary(frame_count,
                        stats,
                        &recorder,
                        &capture_wait_timing,
                        &capture_interval_timing,
                        &yolo_prep_timing,
                        &yolo_npu_timing,
                        &retina_npu_timing,
                        &retina_draw_timing,
                        &processing_timing,
                        &yolo_rknn_perf_timing,
                        &retina_rknn_perf_timing,
                        config.rknn_perf,
                        capture_wait_total_ms,
                        capture_interval_total_ms,
                        clone_total_ms);
    printf("Finished! Last saved clip path: '%s'\n", config.output_path.c_str());
    return 0;
}
