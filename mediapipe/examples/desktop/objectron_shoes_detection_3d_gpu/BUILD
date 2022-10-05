# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("linux_so_library.bzl", "linux_so_library")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//examples:__pkg__"],
)

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_library(
    name = "shoes_detection_lib",
    srcs = ["shoes_detection_lib.cpp"],
    hdrs = ["shoes_detection_lib.h"],
    deps = [
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:parse",
        "//mediapipe/calculators/tflite:tflite_model_calculator",
        "//mediapipe/calculators/util:local_file_contents_calculator",

        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/modules/objectron:objectron_gpu_extended",
    ],
)

cc_binary(
    name = "shoes_detection_gpu",
    srcs = ["shoes_detection_gpu.cpp"],
    deps = [
        ":shoes_detection_lib",
    ],
    linkstatic=False,
)