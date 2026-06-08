// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <string>
#include <algorithm> // Needed for std::max / std::min
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
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

static void yolo_worker(rknn_app_context_t *yolo_ctx,
                        std::deque<FramePacket> *input_queue,
                        std::deque<FramePacket> *retina_queue,
                        std::mutex *queue_mutex,
                        std::condition_variable *queue_cv,
                        std::atomic<bool> *capture_done,
                        std::atomic<bool> *stop_requested,
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
                          std::map<size_t, FramePacket> *finished_frames,
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
            std::lock_guard<std::mutex> lock(*queue_mutex);
            (*finished_frames)[packet.seq] = std::move(packet);
        }
        queue_cv->notify_all();
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("Usage: %s <yolo_model> <retina_model> <video_path>\n", argv[0]);
        return -1;
    }

    const char *yolo_path = argv[1];
    const char *retina_path = argv[2];
    const char *video_path = argv[3];
    int ret;

    init_post_process();

    std::string pipeline = "filesrc location=" + std::string(video_path) + 
                           " ! qtdemux ! h264parse ! mppvideodec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        printf("Failed to open hardware video pipeline!\n");
        return -1;
    }

    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::deque<FramePacket> yolo_queue;
    std::deque<FramePacket> retina_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> capture_done(false);
    std::atomic<bool> stop_requested(false);
    std::map<size_t, FramePacket> finished_frames;
    const size_t max_yolo_queue_size = 4;
    const size_t max_retina_queue_size = 8;
    const int yolo_worker_count = 3;
    const int retina_worker_count = 3;
    const int max_frames = 100;

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
                                    &finished_frames,
                                    i);
    }

    cv::Mat bgr_frame;
    int frame_count = 0;

    printf("--- Starting Real-Time Cascaded Inference Loop ---\n");

    double source_read_total_ms = 0.0;
    double clone_total_ms = 0.0;
    while (true) {
        if (stop_requested.load() || frame_count >= max_frames) {
            break;
        }

        double source_read_start_ms = get_current_time_ms();
        if (!cap.read(bgr_frame)) {
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
                printf("%s[drop][capture][frame-%zu] yolo backlog full, dropping oldest frame%s\n",
                       LOG_YELLOW, dropped.seq, LOG_RESET);
            }
            yolo_queue.push_back(std::move(packet));
        }
        queue_cv.notify_all();
        frame_count++;
    }

    capture_done.store(true);
    queue_cv.notify_all();

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

    std::vector<size_t> ordered_keys;
    ordered_keys.reserve(finished_frames.size());
    for (const auto &entry : finished_frames) {
        ordered_keys.push_back(entry.first);
    }
    std::sort(ordered_keys.begin(), ordered_keys.end());

    double repack_start_ms = get_current_time_ms();
    std::string out_pipeline = "appsrc ! videoconvert ! video/x-raw,format=NV12 ! mpph264enc ! h264parse ! mp4mux ! filesink location=output_cascaded.mp4";
    cv::VideoWriter writer(out_pipeline, cv::CAP_GSTREAMER, 0, 10, cv::Size(w, h));
    for (size_t seq : ordered_keys) {
        writer.write(finished_frames[seq].bgr_frame);
    }
    double repack_ms = get_current_time_ms() - repack_start_ms;
    log_stage_time(LOG_YELLOW, "repack", ordered_keys.size(), 0, 0, repack_ms);

    cap.release();
    writer.release();
    deinit_post_process();
        printf("%s[summary] captured=%d finished=%zu source_read=%.2f ms clone=%.2f ms repack=%.2f ms%s\n",
            LOG_CYAN, frame_count, finished_frames.size(), source_read_total_ms, clone_total_ms, repack_ms, LOG_RESET);
    printf("Finished! Processed %d frames. Saved to 'output_cascaded.mp4'\n", frame_count);
    return 0;
}