// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include "app_config.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool parse_int_arg(const char *value, int *out)
{
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out = (int)parsed;
    return true;
}

static bool parse_size_arg(const char *value, size_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out = (size_t)parsed;
    return true;
}

static bool parse_double_arg(const char *value, double *out)
{
    char *end = NULL;
    errno = 0;
    double parsed = strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out = parsed;
    return true;
}

void print_usage(const char *program)
{
    printf("Usage: %s <yolo_model> <retina_model> [--source rtsp|udp] [--rtsp-url URL] [--rtsp-transport tcp|udp] [--rtsp-latency-ms 200] [--camera-codec h264|h265] [--udp-port 5000] [--clip-frames 150] [--max-buffered-frames 300] [--fps 25] [--output output_cascaded.mp4] [--yolo-workers 1] [--retina-workers 1] [--max-yolo-queue 1] [--max-retina-queue 2] [--yolo-interval 1] [--retina-reinfer-interval 1] [--min-retina-crop 64] [--max-retina-crops 4] [--videoconvert-threads 4] [--npu-core-mode balanced|yolo-all|round-robin] [--rknn-perf] [--async-retina] [--serve-rtsp] [--stream-port 8554] [--stream-path /ai] [--stream-fps 15] [--stream-bitrate-kbps 2500] [--stream-encoder auto|mpp|x264]\n", program);
}

bool parse_config(int argc, char **argv, AppConfig *config)
{
    if (argc < 3) {
        return false;
    }

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
            config->source = argv[++i];
        } else if (strcmp(argv[i], "--rtsp-url") == 0 && i + 1 < argc) {
            config->rtsp_url = argv[++i];
            config->source = "rtsp";
        } else if (strcmp(argv[i], "--rtsp-transport") == 0 && i + 1 < argc) {
            config->rtsp_transport = argv[++i];
            config->source = "rtsp";
        } else if (strcmp(argv[i], "--rtsp-latency-ms") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->rtsp_latency_ms)) {
                return false;
            }
            config->source = "rtsp";
        } else if (strcmp(argv[i], "--camera-codec") == 0 && i + 1 < argc) {
            config->camera_codec = argv[++i];
        } else if (strcmp(argv[i], "--udp-port") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->udp_port)) {
                return false;
            }
            config->source = "udp";
        } else if (strcmp(argv[i], "--clip-frames") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->clip_frames)) {
                return false;
            }
        } else if (strcmp(argv[i], "--max-buffered-frames") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->max_buffered_frames)) {
                return false;
            }
        } else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            if (!parse_double_arg(argv[++i], &config->output_fps)) {
                return false;
            }
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            config->output_path = argv[++i];
        } else if (strcmp(argv[i], "--yolo-workers") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->yolo_worker_count)) {
                return false;
            }
        } else if (strcmp(argv[i], "--retina-workers") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->retina_worker_count)) {
                return false;
            }
        } else if (strcmp(argv[i], "--max-yolo-queue") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->max_yolo_queue_size)) {
                return false;
            }
        } else if (strcmp(argv[i], "--max-retina-queue") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->max_retina_queue_size)) {
                return false;
            }
        } else if (strcmp(argv[i], "--yolo-interval") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->yolo_interval)) {
                return false;
            }
        } else if (strcmp(argv[i], "--retina-reinfer-interval") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->retina_reinfer_interval)) {
                return false;
            }
        } else if (strcmp(argv[i], "--min-retina-crop") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->min_retina_crop_size)) {
                return false;
            }
        } else if (strcmp(argv[i], "--max-retina-crops") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->max_retina_crops)) {
                return false;
            }
        } else if (strcmp(argv[i], "--videoconvert-threads") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->videoconvert_threads)) {
                return false;
            }
        } else if (strcmp(argv[i], "--npu-core-mode") == 0 && i + 1 < argc) {
            config->npu_core_mode = argv[++i];
        } else if (strcmp(argv[i], "--rknn-perf") == 0) {
            config->rknn_perf = true;
        } else if (strcmp(argv[i], "--async-retina") == 0) {
            config->async_retina = true;
        } else if (strcmp(argv[i], "--serve-rtsp") == 0) {
            config->stream_enabled = true;
        } else if (strcmp(argv[i], "--stream-port") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->stream_port)) {
                return false;
            }
            config->stream_enabled = true;
        } else if (strcmp(argv[i], "--stream-path") == 0 && i + 1 < argc) {
            config->stream_path = argv[++i];
            config->stream_enabled = true;
        } else if (strcmp(argv[i], "--stream-fps") == 0 && i + 1 < argc) {
            if (!parse_double_arg(argv[++i], &config->stream_fps)) {
                return false;
            }
            config->stream_enabled = true;
        } else if (strcmp(argv[i], "--stream-bitrate-kbps") == 0 && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], &config->stream_bitrate_kbps)) {
                return false;
            }
            config->stream_enabled = true;
        } else if (strcmp(argv[i], "--stream-encoder") == 0 && i + 1 < argc) {
            config->stream_encoder = argv[++i];
            config->stream_enabled = true;
        } else {
            return false;
        }
    }

    if (config->source != "rtsp" && config->source != "udp") {
        printf("--source must be 'rtsp' or 'udp'\n");
        return false;
    }
    if (config->source == "rtsp" && config->rtsp_url.empty()) {
        printf("--rtsp-url must not be empty when --source rtsp is used\n");
        return false;
    }
    if (config->rtsp_transport != "tcp" && config->rtsp_transport != "udp") {
        printf("--rtsp-transport must be 'tcp' or 'udp'\n");
        return false;
    }
    if (config->rtsp_latency_ms < 0) {
        printf("--rtsp-latency-ms must be >= 0\n");
        return false;
    }
    if (config->camera_codec != "h264" && config->camera_codec != "h265") {
        printf("--camera-codec must be 'h264' or 'h265'\n");
        return false;
    }
    if (config->udp_port <= 0 || config->udp_port > 65535) {
        printf("Invalid --udp-port: %d\n", config->udp_port);
        return false;
    }
    if (config->clip_frames == 0) {
        printf("--clip-frames must be greater than 0\n");
        return false;
    }
    if (config->max_buffered_frames == 0) {
        printf("--max-buffered-frames must be greater than 0\n");
        return false;
    }
    if (config->clip_frames > config->max_buffered_frames) {
        printf("--clip-frames (%zu) must be <= --max-buffered-frames (%zu)\n",
               config->clip_frames, config->max_buffered_frames);
        return false;
    }
    if (config->output_fps <= 0.0) {
        printf("--fps must be greater than 0\n");
        return false;
    }
    if (config->output_path.empty()) {
        printf("--output must not be empty\n");
        return false;
    }
    if (config->yolo_worker_count <= 0) {
        printf("--yolo-workers must be greater than 0\n");
        return false;
    }
    if (config->retina_worker_count <= 0) {
        printf("--retina-workers must be greater than 0\n");
        return false;
    }
    if (config->max_yolo_queue_size == 0) {
        printf("--max-yolo-queue must be greater than 0\n");
        return false;
    }
    if (config->max_retina_queue_size == 0) {
        printf("--max-retina-queue must be greater than 0\n");
        return false;
    }
    if (config->yolo_interval <= 0) {
        printf("--yolo-interval must be greater than 0\n");
        return false;
    }
    if (config->retina_reinfer_interval <= 0) {
        printf("--retina-reinfer-interval must be greater than 0\n");
        return false;
    }
    if (config->min_retina_crop_size < 0) {
        printf("--min-retina-crop must be >= 0\n");
        return false;
    }
    if (config->max_retina_crops < 0) {
        printf("--max-retina-crops must be >= 0\n");
        return false;
    }
    if (config->videoconvert_threads <= 0) {
        printf("--videoconvert-threads must be greater than 0\n");
        return false;
    }
    if (config->npu_core_mode != "balanced" &&
        config->npu_core_mode != "yolo-all" &&
        config->npu_core_mode != "round-robin") {
        printf("--npu-core-mode must be 'balanced', 'yolo-all', or 'round-robin'\n");
        return false;
    }
    if (config->stream_enabled) {
        if (config->stream_port <= 0 || config->stream_port > 65535) {
            printf("Invalid --stream-port: %d\n", config->stream_port);
            return false;
        }
        if (config->stream_path.empty()) {
            printf("--stream-path must not be empty\n");
            return false;
        }
        if (config->stream_fps <= 0.0) {
            printf("--stream-fps must be greater than 0\n");
            return false;
        }
        if (config->stream_bitrate_kbps <= 0) {
            printf("--stream-bitrate-kbps must be greater than 0\n");
            return false;
        }
        if (config->stream_encoder != "auto" &&
            config->stream_encoder != "mpp" &&
            config->stream_encoder != "x264") {
            printf("--stream-encoder must be 'auto', 'mpp', or 'x264'\n");
            return false;
        }
    }
    return true;
}

static std::string decode_output_queue()
{
    return " ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0";
}

static std::string bgr_convert(const AppConfig &config)
{
    return " ! videoconvert n-threads=" + std::to_string(config.videoconvert_threads) +
           " dither=none ! video/x-raw,format=BGR";
}

std::string build_input_pipeline(const AppConfig &config)
{
    std::string depay = "rtph264depay";
    std::string parse = "h264parse";
    if (config.camera_codec == "h265") {
        depay = "rtph265depay";
        parse = "h265parse";
    }

    if (config.source == "udp") {
        return "udpsrc port=" + std::to_string(config.udp_port) +
               " caps=\"video/mpegts, systemstream=(boolean)true, packetsize=(int)188\""
               " ! tsdemux"
               " ! " + parse +
               " ! mppvideodec" + decode_output_queue() +
               bgr_convert(config) +
               " ! appsink drop=true max-buffers=1 sync=false";
    }

    return "rtspsrc location=\"" + config.rtsp_url + "\" latency=" + std::to_string(config.rtsp_latency_ms) +
           " drop-on-latency=true protocols=" + config.rtsp_transport +
           " ! " + depay +
           " ! " + parse +
           " ! mppvideodec" + decode_output_queue() +
           bgr_convert(config) +
           " ! appsink drop=true max-buffers=1 sync=false";
}
