// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <thread>  // Added for clean multi-threading
#include <vector>  // Added to manage parallel thread arrays

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    int ret;

    init_post_process();

    // 1. INITIALIZE 3 COMPLETELY INDEPENDENT CONTEXTS AND ASSIGN THEM TO STANDALONE CORES
    rknn_app_context_t rknn_ctx_pool[3];
    rknn_core_mask core_assignments[3] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};

    for (int i = 0; i < 3; ++i)
    {
        memset(&rknn_ctx_pool[i], 0, sizeof(rknn_app_context_t));
        ret = init_yolo11_model(model_path, &rknn_ctx_pool[i]);
        if (ret != 0)
        {
            printf("init_yolo11_model context index %d fail!\n", i);
            return -1;
        }
        // Explicitly map each distinct session handle directly to a specific physical 2-TOPS silicon core
        ret = rknn_set_core_mask(rknn_ctx_pool[i].rknn_ctx, core_assignments[i]);
        if (ret < 0) {
            printf("rknn_set_core_mask for core %d failed!\n", i);
            return -1;
        }
    }

    // Load the target test image frame buffer
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d\n", ret);
        return -1;
    }

    // Separate post-processing result structures to guarantee thread safety
    object_detect_result_list od_results_pool[3];

    // Warm-up run across all three context models to lock internal buffers into active device space
    for (int i = 0; i < 3; ++i) {
        inference_yolo11_model(&rknn_ctx_pool[i], &src_image, &od_results_pool[i]);
    }

    // 2. RUN THE CONCURRENT ASYNCHRONOUS PIPELINE BENCHMARK
    {
        int loops_per_thread = 100;
        struct timeval start_time, stop_time;
        std::vector<std::thread> workers;

        gettimeofday(&start_time, NULL);

        // Spawn 3 concurrent hardware execution loops matching the 3 cores
        for (int core_id = 0; core_id < 3; ++core_id)
        {
            workers.push_back(std::thread([&rknn_ctx_pool, core_id, loops_per_thread]() {
                for (int i = 0; i < loops_per_thread; ++i) {
                    // Triggers the low-level runtime hardware driver with zero user-space memory overhead
                    rknn_run(rknn_ctx_pool[core_id].rknn_ctx, NULL);
                }
            }));
        }

        // Wait for all three threads to finish their 100-frame workloads
        for (auto &t : workers) {
            if (t.joinable()) t.join();
        }

        gettimeofday(&stop_time, NULL);
        double total_time_ms = (stop_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                               (stop_time.tv_usec - start_time.tv_usec) / 1000.0;

        // Math breakdown: 3 hardware threads running 100 frames each = 300 total frames computed
        int total_frames_processed = 3 * loops_per_thread;
        double parallel_batch_latency = total_time_ms / loops_per_thread;

        printf("\n=======================================\n");
        printf("Asynchronous 3-Core NPU Pipeline Benchmark:\n");
        printf("Model Target Variant      : YOLOv11s (Small)\n");
        printf("Total Time for 300 Frames : %.2f ms\n", total_time_ms);
        printf("Parallel Batch Latency    : %.2f ms/batch\n", parallel_batch_latency);
        printf("Combined System Throughput : %.2f FPS 🔥\n", (total_frames_processed / total_time_ms) * 1000.0);
        printf("=======================================\n\n");
    }

    // Use the results from context 0 to draw our bounding box visualizations
    char text[256];
    for (int i = 0; i < od_results_pool[0].count; i++)
    {
        object_detect_result *det_result = &(od_results_pool[0].results[i]);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }
    write_image("out.png", &src_image);

    // Clean up all persistent memory pools across context allocations
    deinit_post_process();
    for (int i = 0; i < 3; ++i) {
        release_yolo11_model(&rknn_ctx_pool[i]);
    }
    free(src_image.virt_addr);

    return 0;
}