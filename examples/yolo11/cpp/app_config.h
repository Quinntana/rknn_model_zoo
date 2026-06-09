// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#ifndef RKNN_YOLO11_APP_CONFIG_H_
#define RKNN_YOLO11_APP_CONFIG_H_

#include <stddef.h>

#include <string>

struct AppConfig {
    std::string source = "rtsp";
    std::string rtsp_url = "rtsp://admin:admin@192.168.1.156:554/stream1";
    std::string rtsp_transport = "tcp";
    int rtsp_latency_ms = 200;
    std::string camera_codec = "h264";
    int udp_port = 5000;
    size_t clip_frames = 150;
    size_t max_buffered_frames = 300;
    double output_fps = 25.0;
    std::string output_path = "output_cascaded.mp4";
    int yolo_worker_count = 1;
    int retina_worker_count = 1;
    size_t max_yolo_queue_size = 1;
    size_t max_retina_queue_size = 2;
    int yolo_interval = 1;
    int min_retina_crop_size = 64;
    int max_retina_crops = 4;
    int videoconvert_threads = 4;
    int retina_reinfer_interval = 1;
    std::string npu_core_mode = "balanced";
    bool rknn_perf = false;
    bool async_retina = false;
};

void print_usage(const char *program);
bool parse_config(int argc, char **argv, AppConfig *config);
std::string build_input_pipeline(const AppConfig &config);

#endif  // RKNN_YOLO11_APP_CONFIG_H_
