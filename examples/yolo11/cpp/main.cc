// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <string>

// Include OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"

double get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s <model_path> <video_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *video_path = argv[2];
    int ret;

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    init_post_process();

    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0) return -1;
    
    // Pin to Core 0
    rknn_set_core_mask(rknn_app_ctx.rknn_ctx, RKNN_NPU_CORE_0);

    std::string pipeline = "filesrc location=" + std::string(video_path) + 
                           " ! qtdemux ! h264parse ! mppvideodec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        printf("Failed to open hardware video pipeline!\n");
        return -1;
    }

    // --- Initialize the Video Writer ---
    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer("output_hardware.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(w, h));

    cv::Mat bgr_frame, rgb_frame;
    object_detect_result_list od_results;
    int frame_count = 0;

    printf("--- Starting Real-Time ALPR Inference Loop ---\n");

    while (cap.read(bgr_frame)) {
        // [TIMING STAGE 0] Pipeline Start
        double t0_start = get_current_time_ms();

        // -----------------------------------------
        // STAGE 1: Pre-processing (Video Frame Prep)
        // -----------------------------------------
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
        
        // [TIMING STAGE 1] Pre-processing complete
        double t1_prep = get_current_time_ms();

        // -----------------------------------------
        // STAGE 2: AI Detection (YOLO Vehicle/Plate)
        // -----------------------------------------
        ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0) break;
        
        // [TIMING STAGE 2] YOLO Detection complete
        double t2_detect = get_current_time_ms();

        // -----------------------------------------
        // STAGE 3: AI OCR (FUTURE ALPR INTEGRATION)
        // -----------------------------------------
        // TODO: Loop through od_results. If class == "license_plate", 
        // crop the bounding box from bgr_frame and pass to the OCR model.
        
        // [TIMING STAGE 3] OCR complete
        double t3_ocr = get_current_time_ms(); 

        // -----------------------------------------
        // STAGE 4: Post-processing (Draw & Encode)
        // -----------------------------------------
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            cv::rectangle(bgr_frame, cv::Point(det_result->box.left, det_result->box.top), 
                          cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 255, 0), 2);
            char text[256];
            // TODO: Append OCR text results to this string in the future
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            cv::putText(bgr_frame, text, cv::Point(det_result->box.left, det_result->box.top - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }

        writer.write(bgr_frame);
        
        // [TIMING STAGE 4] Post-processing & File I/O complete
        double t4_end = get_current_time_ms();

        // -----------------------------------------
        // PIPELINE METRICS REPORTING
        // -----------------------------------------
        if (frame_count % 30 == 0) {
            double prep_time = t1_prep - t0_start;
            double detect_time = t2_detect - t1_prep;
            double ocr_time = t3_ocr - t2_detect;
            double post_time = t4_end - t3_ocr;
            double total_time = t4_end - t0_start;
            
            printf("Frame %4d | Total: %5.1fms | FPS: %4.1f | Profiling -> Prep: %4.1fms | Det: %4.1fms | OCR: %4.1fms | Post: %4.1fms\n", 
                   frame_count, total_time, 1000.0 / total_time, prep_time, detect_time, ocr_time, post_time);
        }
        frame_count++;
    }

    // Clean up
    cap.release();
    writer.release();
    deinit_post_process();
    release_yolo11_model(&rknn_app_ctx);
    
    printf("\nFinished! Processed %d frames. Saved to 'output_hardware.mp4'\n", frame_count);
    return 0;
}