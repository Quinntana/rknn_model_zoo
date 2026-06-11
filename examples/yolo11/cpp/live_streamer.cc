// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include "live_streamer.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

#include "runtime_state.h"

LiveStreamer::LiveStreamer(const AppConfig &config)
    : enabled_(config.stream_enabled),
      port_(config.stream_port),
      mount_path_(normalize_mount_path(config.stream_path)),
      fps_(config.stream_fps),
      bitrate_kbps_(config.stream_bitrate_kbps),
      encoder_preference_(config.stream_encoder)
{
    double rounded_fps = std::round(fps_);
    if (std::fabs(fps_ - rounded_fps) < 0.001) {
        fps_num_ = std::max(1, static_cast<int>(rounded_fps));
        fps_den_ = 1;
    } else {
        fps_num_ = std::max(1, static_cast<int>(std::round(fps_ * 1000.0)));
        fps_den_ = 1000;
    }
}

LiveStreamer::~LiveStreamer()
{
    stop();
}

bool LiveStreamer::enabled() const
{
    return enabled_;
}

void LiveStreamer::publish(const cv::Mat &bgr_frame)
{
    if (!enabled_ || stop_requested_.load()) {
        return;
    }
    if (bgr_frame.empty() || bgr_frame.type() != CV_8UC3) {
        return;
    }
    if (!ensure_started(bgr_frame.cols, bgr_frame.rows)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (bgr_frame.cols != frame_width_ || bgr_frame.rows != frame_height_) {
            if (!size_warning_printed_) {
                printf("%s[stream] dropping frame with changed size %dx%d; expected %dx%d%s\n",
                       LOG_YELLOW,
                       bgr_frame.cols,
                       bgr_frame.rows,
                       frame_width_,
                       frame_height_,
                       LOG_RESET);
                size_warning_printed_ = true;
            }
            return;
        }

        latest_frame_ = bgr_frame.clone();
        latest_seq_++;
        latest_valid_ = true;
    }
    frame_cv_.notify_one();
}

void LiveStreamer::stop()
{
    if (!enabled_) {
        return;
    }

    stop_requested_.store(true);
    frame_cv_.notify_all();

    if (push_thread_.joinable()) {
        push_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(appsrc_mutex_);
        if (appsrc_ != NULL) {
            gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
            gst_object_unref(appsrc_);
            appsrc_ = NULL;
        }
    }

    if (main_loop_ != NULL) {
        g_main_loop_quit(main_loop_);
    }
    if (main_loop_thread_.joinable()) {
        main_loop_thread_.join();
    }

    std::lock_guard<std::mutex> lock(state_mutex_);
    if (source_id_ != 0) {
        g_source_remove(source_id_);
        source_id_ = 0;
    }
    if (factory_ != NULL) {
        gst_object_unref(factory_);
        factory_ = NULL;
    }
    if (server_ != NULL) {
        gst_object_unref(server_);
        server_ = NULL;
    }
    if (main_loop_ != NULL) {
        g_main_loop_unref(main_loop_);
        main_loop_ = NULL;
    }
    started_ = false;
}

bool LiveStreamer::ensure_started(int width, int height)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (started_) {
        return true;
    }
    if (start_failed_ || stop_requested_.load()) {
        return false;
    }

    GError *error = NULL;
    if (!gst_init_check(NULL, NULL, &error)) {
        printf("%s[stream] failed to initialize GStreamer: %s%s\n",
               LOG_RED,
               error != NULL ? error->message : "unknown error",
               LOG_RESET);
        if (error != NULL) {
            g_error_free(error);
        }
        start_failed_ = true;
        return false;
    }

    frame_width_ = width;
    frame_height_ = height;
    selected_encoder_ = select_encoder();
    if (selected_encoder_.empty()) {
        start_failed_ = true;
        return false;
    }

    main_loop_ = g_main_loop_new(NULL, FALSE);
    server_ = gst_rtsp_server_new();
    factory_ = gst_rtsp_media_factory_new();
    if (main_loop_ == NULL || server_ == NULL || factory_ == NULL) {
        printf("%s[stream] failed to allocate RTSP server objects%s\n", LOG_RED, LOG_RESET);
        start_failed_ = true;
        return false;
    }

    std::string service = std::to_string(port_);
    g_object_set(server_, "address", "0.0.0.0", "service", service.c_str(), NULL);

    std::string launch = launch_pipeline();
    gst_rtsp_media_factory_set_launch(factory_, launch.c_str());
    gst_rtsp_media_factory_set_shared(factory_, TRUE);
    g_signal_connect(factory_, "media-configure", G_CALLBACK(LiveStreamer::on_media_configure), this);

    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server_);
    gst_rtsp_mount_points_add_factory(mounts, mount_path_.c_str(), factory_);
    gst_object_ref(factory_);
    g_object_unref(mounts);

    source_id_ = gst_rtsp_server_attach(server_, NULL);
    if (source_id_ == 0) {
        printf("%s[stream] failed to bind RTSP server on port %d%s\n", LOG_RED, port_, LOG_RESET);
        start_failed_ = true;
        return false;
    }

    main_loop_thread_ = std::thread([this]() {
        g_main_loop_run(main_loop_);
    });
    push_thread_ = std::thread(&LiveStreamer::push_loop, this);
    started_ = true;

    printf("%s[stream] live AI RTSP ready on 0.0.0.0:%d%s fps=%.2f bitrate=%d kbps encoder=%s%s\n",
           LOG_CYAN,
           port_,
           mount_path_.c_str(),
           fps_,
           bitrate_kbps_,
           selected_encoder_.c_str(),
           LOG_RESET);
    std::vector<std::string> urls = stream_urls();
    for (const std::string &url : urls) {
        printf("%s[stream] open this URL from VLC on the laptop: %s%s\n",
               LOG_CYAN,
               url.c_str(),
               LOG_RESET);
    }
    return true;
}

void LiveStreamer::push_loop()
{
    const uint64_t duration_ns = static_cast<uint64_t>(1000000000.0 / fps_);
    const auto frame_interval = std::chrono::nanoseconds(duration_ns);
    auto next_push = std::chrono::steady_clock::now();
    uint64_t frame_index = 0;

    while (!stop_requested_.load()) {
        if (!has_active_appsrc()) {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                return stop_requested_.load();
            });
            next_push = std::chrono::steady_clock::now();
            continue;
        }

        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait(lock, [this]() {
                return latest_valid_ || stop_requested_.load();
            });
            if (stop_requested_.load()) {
                break;
            }
            frame = latest_frame_.clone();
        }

        push_frame(frame, frame_index++, duration_ns);

        next_push += frame_interval;
        std::unique_lock<std::mutex> lock(frame_mutex_);
        frame_cv_.wait_until(lock, next_push, [this]() {
            return stop_requested_.load();
        });
    }
}

void LiveStreamer::push_frame(const cv::Mat &bgr_frame, uint64_t frame_index, uint64_t duration_ns)
{
    GstElement *appsrc = NULL;
    {
        std::lock_guard<std::mutex> lock(appsrc_mutex_);
        if (appsrc_ != NULL) {
            appsrc = GST_ELEMENT(gst_object_ref(appsrc_));
        }
    }
    if (appsrc == NULL) {
        return;
    }

    cv::Mat contiguous = bgr_frame.isContinuous() ? bgr_frame : bgr_frame.clone();
    const size_t frame_size = contiguous.total() * contiguous.elemSize();
    GstBuffer *buffer = gst_buffer_new_allocate(NULL, frame_size, NULL);
    if (buffer == NULL) {
        gst_object_unref(appsrc);
        return;
    }

    gst_buffer_fill(buffer, 0, contiguous.data, frame_size);
    GST_BUFFER_PTS(buffer) = frame_index * duration_ns;
    GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
    GST_BUFFER_DURATION(buffer) = duration_ns;
    GST_BUFFER_OFFSET(buffer) = frame_index;

    GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    if (flow != GST_FLOW_OK && flow != GST_FLOW_FLUSHING && flow != GST_FLOW_EOS) {
        printf("%s[stream] appsrc push failed: flow=%d%s\n", LOG_YELLOW, flow, LOG_RESET);
    }
    gst_object_unref(appsrc);
}

bool LiveStreamer::has_active_appsrc()
{
    std::lock_guard<std::mutex> lock(appsrc_mutex_);
    return appsrc_ != NULL;
}

std::string LiveStreamer::select_encoder() const
{
    if (encoder_preference_ == "mpp" || encoder_preference_ == "auto") {
        if (gst_element_available("mpph264enc")) {
            printf("%s[stream] using Rockchip hardware encoder: mpph264enc%s\n",
                   LOG_CYAN,
                   LOG_RESET);
            return "mpp";
        }

        if (encoder_preference_ == "mpp") {
            printf("%s[stream] --stream-encoder mpp requested, but GStreamer element mpph264enc is not available%s\n",
                   LOG_RED,
                   LOG_RESET);
            return "";
        }

        printf("%s[stream] mpph264enc not available; falling back to x264enc%s\n",
               LOG_YELLOW,
               LOG_RESET);
    }

    if (!gst_element_available("x264enc")) {
        printf("%s[stream] GStreamer element x264enc is not available%s\n",
               LOG_RED,
               LOG_RESET);
        return "";
    }

    return "x264";
}

std::string LiveStreamer::encoder_pipeline() const
{
    if (selected_encoder_ == "mpp") {
        return " ! videoconvert"
               " ! video/x-raw,format=NV12"
               " ! mpph264enc"
               " ! h264parse config-interval=1";
    }

    int key_int_max = std::max(1, static_cast<int>(std::round(fps_)));
    std::ostringstream out;
    out << " ! videoconvert"
        << " ! video/x-raw,format=I420"
        << " ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=" << bitrate_kbps_
        << " key-int-max=" << key_int_max
        << " bframes=0"
        << " ! video/x-h264,profile=baseline"
        << " ! h264parse config-interval=1";
    return out.str();
}

std::string LiveStreamer::launch_pipeline() const
{
    std::ostringstream out;
    out << "( appsrc name=ai_src is-live=true block=false format=time do-timestamp=false "
        << "caps=video/x-raw,format=BGR,width=" << frame_width_
        << ",height=" << frame_height_
        << ",framerate=" << fps_num_ << "/" << fps_den_
        << " ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 "
        << encoder_pipeline()
        << " ! rtph264pay name=pay0 pt=96 config-interval=1 )";
    return out.str();
}

std::string LiveStreamer::stream_url() const
{
    std::vector<std::string> urls = stream_urls();
    if (!urls.empty()) {
        return urls[0];
    }

    std::ostringstream out;
    out << "rtsp://127.0.0.1:" << port_ << mount_path_;
    return out.str();
}

std::vector<std::string> LiveStreamer::stream_urls() const
{
    std::vector<std::string> ips = detect_local_ips();
    std::vector<std::string> urls;
    urls.reserve(ips.size());
    for (const std::string &ip : ips) {
        std::ostringstream out;
        out << "rtsp://" << ip << ":" << port_ << mount_path_;
        urls.push_back(out.str());
    }
    return urls;
}

std::string LiveStreamer::normalize_mount_path(const std::string &path)
{
    if (path.empty()) {
        return "/ai";
    }
    if (path[0] == '/') {
        return path;
    }
    return "/" + path;
}

bool LiveStreamer::gst_element_available(const char *factory_name)
{
    GstElementFactory *factory = gst_element_factory_find(factory_name);
    if (factory == NULL) {
        return false;
    }
    gst_object_unref(factory);
    return true;
}

std::vector<std::string> LiveStreamer::detect_local_ips()
{
    struct ifaddrs *ifaddr = NULL;
    if (getifaddrs(&ifaddr) != 0) {
        return std::vector<std::string>{"127.0.0.1"};
    }

    std::vector<std::string> results;
    for (struct ifaddrs *ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_addr->sa_family != AF_INET) {
            continue;
        }
        if ((ifa->ifa_flags & IFF_LOOPBACK) != 0 || (ifa->ifa_flags & IFF_UP) == 0) {
            continue;
        }

        char host[INET_ADDRSTRLEN] = {0};
        struct sockaddr_in *addr = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
        if (inet_ntop(AF_INET, &addr->sin_addr, host, sizeof(host)) != NULL) {
            results.push_back(host);
        }
    }

    freeifaddrs(ifaddr);
    if (results.empty()) {
        results.push_back("127.0.0.1");
    }
    return results;
}

void LiveStreamer::on_media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data)
{
    (void)factory;
    LiveStreamer *self = static_cast<LiveStreamer *>(user_data);
    GstElement *element = gst_rtsp_media_get_element(media);
    if (element == NULL) {
        return;
    }

    GstElement *appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(element), "ai_src");
    gst_object_unref(element);
    if (appsrc == NULL) {
        printf("%s[stream] media configured without appsrc%s\n", LOG_RED, LOG_RESET);
        return;
    }

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                        "format", G_TYPE_STRING, "BGR",
                                        "width", G_TYPE_INT, self->frame_width_,
                                        "height", G_TYPE_INT, self->frame_height_,
                                        "framerate", GST_TYPE_FRACTION, self->fps_num_, self->fps_den_,
                                        NULL);
    gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
    gst_caps_unref(caps);

    g_object_set(appsrc,
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "block", FALSE,
                 "max-buffers", static_cast<guint64>(1),
                 "max-bytes", static_cast<guint64>(0),
                 "max-time", static_cast<guint64>(0),
                 "leaky-type", GST_APP_LEAKY_TYPE_DOWNSTREAM,
                 NULL);
    gst_app_src_set_stream_type(GST_APP_SRC(appsrc), GST_APP_STREAM_TYPE_STREAM);
    gst_app_src_set_latency(GST_APP_SRC(appsrc), 0, GST_CLOCK_TIME_NONE);

    {
        std::lock_guard<std::mutex> lock(self->appsrc_mutex_);
        if (self->appsrc_ != NULL) {
            gst_object_unref(self->appsrc_);
        }
        self->appsrc_ = appsrc;
    }
    g_signal_connect(media, "unprepared", G_CALLBACK(LiveStreamer::on_media_unprepared), self);

    printf("%s[stream] VLC client connected to %s%s\n",
           LOG_CYAN,
           self->stream_url().c_str(),
           LOG_RESET);
}

void LiveStreamer::on_media_unprepared(GstRTSPMedia *media, gpointer user_data)
{
    (void)media;
    LiveStreamer *self = static_cast<LiveStreamer *>(user_data);
    std::lock_guard<std::mutex> lock(self->appsrc_mutex_);
    if (self->appsrc_ != NULL) {
        gst_object_unref(self->appsrc_);
        self->appsrc_ = NULL;
    }
    printf("%s[stream] VLC client disconnected%s\n", LOG_YELLOW, LOG_RESET);
}
