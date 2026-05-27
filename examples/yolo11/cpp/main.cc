// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <string>
#include <algorithm> // Needed for std::max / std::min

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

    // 1. INITIALIZE DUAL CONTEXTS (Notice the strictly separated struct types!)
    rknn_app_context_t yolo_ctx;
    retina_app_context_t retina_ctx; 
    
    memset(&yolo_ctx, 0, sizeof(rknn_app_context_t));
    memset(&retina_ctx, 0, sizeof(retina_app_context_t));
    init_post_process();

    // Init YOLO and allow RK3588 multi-core execution instead of single-core pinning
    ret = init_yolo11_model(yolo_path, &yolo_ctx);
    if (ret != 0) return -1;
    rknn_set_core_mask(yolo_ctx.rknn_ctx, RKNN_NPU_CORE_0_1_2);
    ret = rknn_set_batch_core_num(yolo_ctx.rknn_ctx, 3);
    if (ret != RKNN_SUCC) {
        printf("rknn_set_batch_core_num(yolo) fail! ret=%d\n", ret);
    }

    // Put RetinaFace on the same RK3588 multi-core mode so it can use all NPU cores
    ret = init_retinaface_model(retina_path, &retina_ctx);
    if (ret != 0) return -1;
    rknn_set_core_mask(retina_ctx.rknn_ctx, RKNN_NPU_CORE_0_1_2);
    ret = rknn_set_batch_core_num(retina_ctx.rknn_ctx, 3);
    if (ret != RKNN_SUCC) {
        printf("rknn_set_batch_core_num(retina) fail! ret=%d\n", ret);
    }

    std::string pipeline = "filesrc location=" + std::string(video_path) + 
                           " ! qtdemux ! h264parse ! mppvideodec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        printf("Failed to open hardware video pipeline!\n");
        return -1;
    }

    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::string out_pipeline = "appsrc ! videoconvert ! video/x-raw,format=NV12 ! mpph264enc ! h264parse ! mp4mux ! filesink location=output_cascaded.mp4";
    cv::VideoWriter writer(out_pipeline, cv::CAP_GSTREAMER, 0, 30, cv::Size(w, h));

    cv::Mat bgr_frame, rgb_frame;
    object_detect_result_list od_results;
    int frame_count = 0;
    int MAX_FRAMES = 300; // 5 seconds of video for testing

    printf("--- Starting Real-Time Cascaded Inference Loop ---\n");

    while (cap.read(bgr_frame)) {
        if (frame_count >= MAX_FRAMES) break;

        double t0_start = get_current_time_ms();

        cv::cvtColor(bgr_frame, rgb_frame, cv::COLOR_BGR2RGB);

        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.width = rgb_frame.cols;
        src_image.height = rgb_frame.rows;
        src_image.width_stride = rgb_frame.cols;
        src_image.height_stride = rgb_frame.rows;
        src_image.format = IMAGE_FORMAT_RGB888; 
        src_image.virt_addr = rgb_frame.data; 
        src_image.size = rgb_frame.total() * rgb_frame.channels();
        
        double t1_prep = get_current_time_ms();

        ret = inference_yolo11_model(&yolo_ctx, &src_image, &od_results);
        if (ret != 0) break;
        
        double t2_detect = get_current_time_ms();

        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            
            // Only process if YOLO found a Person (COCO Class 0)
            if (det_result->cls_id == 0) {
                
                // 1. ALIGN coordinates to multiples of 4 to keep the RGA hardware accelerator happy (Fixes Performance)
                // Change (~3) to (~15) to guarantee 16-pixel alignment
                int x1 = std::max(0, det_result->box.left) & (~15);
                int y1 = std::max(0, det_result->box.top) & (~15);
                int x2 = std::min(rgb_frame.cols - 1, det_result->box.right);
                int y2 = std::min(rgb_frame.rows - 1, det_result->box.bottom);

                int crop_w = (x2 - x1) & (~15); 
                int crop_h = (y2 - y1) & (~15);

                if (crop_w <= 0 || crop_h <= 0) continue;

                // 2. CLONE the crop so the memory is contiguous (Fixes 0 Detections)
                cv::Mat person_crop = rgb_frame(cv::Rect(x1, y1, crop_w, crop_h)).clone();

                image_buffer_t crop_img;
                memset(&crop_img, 0, sizeof(image_buffer_t));
                crop_img.width = person_crop.cols;
                crop_img.height = person_crop.rows;
                crop_img.width_stride = person_crop.cols; // This is now accurate because of .clone()
                crop_img.height_stride = person_crop.rows;
                crop_img.format = IMAGE_FORMAT_RGB888; 
                crop_img.virt_addr = person_crop.data; 
                crop_img.size = person_crop.total() * person_crop.channels();

                retinaface_result retina_res;
                inference_retinaface_model(&retina_ctx, &crop_img, &retina_res);

                for (int f = 0; f < retina_res.count; f++) {
                    if (retina_res.object[f].score > 0.5) { 
                        for (int p = 0; p < 5; p++) {
                            // Coordinate Translation: Map crop points back to global video frame
                            int global_x = x1 + retina_res.object[f].ponit[p].x;
                            int global_y = y1 + retina_res.object[f].ponit[p].y;
                            
                            // Draw Orange circles on facial features
                            cv::circle(bgr_frame, cv::Point(global_x, global_y), 4, cv::Scalar(0, 165, 255), -1);
                        }
                    }
                }
            }
            
            // Draw YOLO Person Box
            cv::rectangle(bgr_frame, cv::Point(det_result->box.left, det_result->box.top), 
                          cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 255, 0), 2);
            char text[256];
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            cv::putText(bgr_frame, text, cv::Point(det_result->box.left, det_result->box.top - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
        
        double t3_ocr = get_current_time_ms(); 

        writer.write(bgr_frame);
        
        double t4_end = get_current_time_ms();

        if (frame_count % 30 == 0) {
            double prep_time = t1_prep - t0_start;
            double detect_time = t2_detect - t1_prep;
            double casc_time = t3_ocr - t2_detect;
            double post_time = t4_end - t3_ocr;
            double total_time = t4_end - t0_start;
            
            printf("Frame %4d | Total: %5.1fms | FPS: %4.1f | Prep: %4.1fms | YOLO: %4.1fms | Retina: %4.1fms | Encode: %4.1fms\n", 
                   frame_count, total_time, 1000.0 / total_time, prep_time, detect_time, casc_time, post_time);
        }
        frame_count++;
    }

    cap.release();
    writer.release();
    deinit_post_process();
    release_yolo11_model(&yolo_ctx);
    release_retinaface_model(&retina_ctx);
    
    printf("\nFinished! Processed %d frames. Saved to 'output_cascaded.mp4'\n", frame_count);
    return 0;
}