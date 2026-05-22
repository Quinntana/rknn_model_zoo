// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h> // Correctly included for gettimeofday

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

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
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolo11_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);

#if defined(RV1106_1103) 
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                       (void **) & (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
    memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
    dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
    free(src_image.virt_addr);
    src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
    src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
    rknn_app_ctx.img_dma_buf.size = src_image.size;
#endif
    
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    object_detect_result_list od_results;

    // --- WARM UP RUN ---
    ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
    if (ret != 0)
    {
        printf("inference_yolo11_model fail! ret=%d\n", ret);
        goto out;
    }

    // --- DUAL BENCHMARK SCOPE ---
    {
        int loop_count = 100;
        struct timeval start_time, stop_time;
        
        // 1. Measure the COMPLETE Pipeline (Pre + Inference + Post)
        gettimeofday(&start_time, NULL);
        for (int i = 0; i < loop_count; ++i) {
            inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
        }
        gettimeofday(&stop_time, NULL);
        double total_time_ms = (stop_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                               (stop_time.tv_usec - start_time.tv_usec) / 1000.0;
        
        // 2. Measure ONLY the Core Hardware NPU Execution
        struct timeval npu_start, npu_stop;
        gettimeofday(&npu_start, NULL);
        for (int i = 0; i < loop_count; ++i) {
            // Directly invokes the low-level runtime driver on the current memory buffer
            rknn_run(rknn_app_ctx.rknn_ctx, NULL);
        }
        gettimeofday(&npu_stop, NULL);
        double npu_time_ms = (npu_stop.tv_sec - npu_start.tv_sec) * 1000.0 + 
                             (npu_stop.tv_usec - npu_start.tv_usec) / 1000.0;
        
        printf("\n=======================================\n");
        printf("Official RKNN Zoo Benchmark (%d loops):\n", loop_count);
        printf("End-to-End Pipeline    : %.2f ms (%.1f FPS)\n", total_time_ms / loop_count, 1000.0 / (total_time_ms / loop_count));
        printf("Pure NPU Inference Only: %.2f ms (%.1f FPS)\n", npu_time_ms / loop_count, 1000.0 / (npu_time_ms / loop_count));
        printf("=======================================\n\n");
    }

    // 画框和概率
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }

    write_image("out.png", &src_image);

out:
    deinit_post_process();

    ret = release_yolo11_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolo11_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
#if defined(RV1106_1103) 
        dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
        free(src_image.virt_addr);
#endif
    }

    return 0;
}