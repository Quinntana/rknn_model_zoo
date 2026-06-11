// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include "worker_threads.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "image_utils.h"
#include "live_streamer.h"

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

bool write_clip_mp4(const std::vector<RecordedFrame> &frames,
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

void input_worker(std::atomic<bool> *stop_requested,
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

void complete_frame(FramePacket *packet,
                    RecorderBuffer *recorder,
                    LiveStreamer *live_streamer,
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

    if (live_streamer != NULL) {
        live_streamer->publish(packet->bgr_frame);
    }
    recorder->add_completed(packet->seq, std::move(packet->bgr_frame));
    stats->processed_frames.fetch_add(1);
    queue_cv->notify_all();
}

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
                           live_streamer,
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
                               live_streamer,
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
                               live_streamer,
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

            if (live_streamer != NULL) {
                live_streamer->publish(packet.bgr_frame);
            }
            recorder->add_completed(packet.seq, std::move(packet.bgr_frame));
            stats->processed_frames.fetch_add(1);
        }
        queue_cv->notify_all();
    }
}
