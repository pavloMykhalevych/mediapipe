# Copyright 2020 The MediaPipe Authors.
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

load("@bazel_skylib//lib:selects.bzl", "selects")
load(":build_defs.bzl", "generate_manifest_values")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "libmediapipe_jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/graphs/object_detection_3d:mobile_calculators_extended",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
    ],
)

cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    alwayslink = 1,
)

cc_binary(
    name = "libmediapipegpu.so",
    linkshared = 1,
    linkstatic = 1,
    srcs = ["shoes_detection_lib.cpp", "shoes_detection_lib.h"],
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

cc_library(
    name = "mediapipe_gpu_lib",
    srcs = [":libmediapipegpu.so"],
    alwayslink = 1,
)

genrule(
    name = "binary_graph",
    srcs = ["//mediapipe/graphs/object_detection_3d:mobile_gpu_binary_graph_modified"],
    outs = ["object_detection_3d.binarypb"],
    cmd = "cp $< $@",
)

MODELS_DIR = "//mediapipe/modules/objectron"

genrule(
    name = "model",
    srcs = [MODELS_DIR + ":object_detection_3d_sneakers.tflite"],
    outs = ["object_detection_3d.tflite"],
    cmd = "cp $< $@",
)

MANIFESTS_DIR = "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/manifests"

android_library(
    name = "manifest_lib",
    exports_manifest = 1,
    manifest =  MANIFESTS_DIR + ":AndroidManifestSneaker.xml",
    deps = [
        "//third_party:opencv",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:com_google_guava_guava",
    ],
)

ASSETS_DIR = "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/assets"

genrule(
    name = "mesh",
    srcs = [ASSETS_DIR + "/sneaker:model.obj.uuu"],
    outs = ["model.obj.uuu"],
    cmd = "cp $< $@",
)

genrule(
    name = "texture",
    srcs = [ASSETS_DIR + "/sneaker:texture.jpg"],
    outs = ["texture.jpg"],
    cmd = "cp $< $@",
)

android_binary(
    name = "objectdetection3dgpu",
    srcs = glob(["*.java"]),
    assets = [
        ":binary_graph",
        ":model",
        ":mesh",
        ":texture",
        MODELS_DIR + ":object_detection_ssd_mobilenetv2_oidv4_fp16.tflite",
        MODELS_DIR + ":object_detection_oidv4_labelmap.txt",
        ASSETS_DIR + ":box.obj.uuu",
        ASSETS_DIR + ":classic_colors.png",
    ],
    assets_dir = "",
    manifest = "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:AndroidManifest.xml",
    manifest_values = generate_manifest_values("com.google.mediapipe.apps.objectdetection3d_shoe_modified", "Shoe Objectron Modified"),
    multidex = "native",
    deps = [
        ":manifest_lib",
        ":mediapipe_jni_lib",
        ":mediapipe_gpu_lib",
        "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:basic_lib",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
    ],
)
