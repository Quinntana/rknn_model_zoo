// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#ifndef RKNN_YOLO11_LIVE_STREAMER_H_
#define RKNN_YOLO11_LIVE_STREAMER_H_

#include <stdint.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <opencv2/core.hpp>

#include "app_config.h"

class LiveStreamer {
public:
    explicit LiveStreamer(const AppConfig &config);
    ~LiveStreamer();

    LiveStreamer(const LiveStreamer &) = delete;
    LiveStreamer &operator=(const LiveStreamer &) = delete;

    bool enabled() const;
    void publish(const cv::Mat &bgr_frame);
    void stop();

private:
    bool ensure_started(int width, int height);
    void push_loop();
    void push_frame(const cv::Mat &bgr_frame, uint64_t frame_index, uint64_t duration_ns);
    bool has_active_appsrc();
    std::string select_encoder() const;
    std::string encoder_pipeline() const;
    std::string launch_pipeline() const;
    std::string stream_url() const;
    std::vector<std::string> stream_urls() const;

    static bool gst_element_available(const char *factory_name);
    static std::string normalize_mount_path(const std::string &path);
    static std::vector<std::string> detect_local_ips();
    static void on_media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data);
    static void on_media_unprepared(GstRTSPMedia *media, gpointer user_data);

    bool enabled_ = false;
    int port_ = 8554;
    std::string mount_path_ = "/ai";
    double fps_ = 15.0;
    int fps_num_ = 15;
    int fps_den_ = 1;
    int bitrate_kbps_ = 2500;
    std::string encoder_preference_ = "auto";
    std::string selected_encoder_;
    int frame_width_ = 0;
    int frame_height_ = 0;

    std::atomic<bool> stop_requested_{false};

    std::mutex state_mutex_;
    bool started_ = false;
    bool start_failed_ = false;
    GMainLoop *main_loop_ = NULL;
    GstRTSPServer *server_ = NULL;
    GstRTSPMediaFactory *factory_ = NULL;
    guint source_id_ = 0;
    std::thread main_loop_thread_;
    std::thread push_thread_;

    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    cv::Mat latest_frame_;
    uint64_t latest_seq_ = 0;
    bool latest_valid_ = false;
    bool size_warning_printed_ = false;

    std::mutex appsrc_mutex_;
    GstElement *appsrc_ = NULL;
};

#endif  // RKNN_YOLO11_LIVE_STREAMER_H_
