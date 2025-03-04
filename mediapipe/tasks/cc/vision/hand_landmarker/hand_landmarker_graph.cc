/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "mediapipe/calculators/core/clip_vector_size_calculator.pb.h"
#include "mediapipe/calculators/core/gate_calculator.pb.h"
#include "mediapipe/calculators/util/collection_has_min_size_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/utils/gate.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_association_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::utils::DisallowIf;
using ::mediapipe::tasks::core::ModelAssetBundleResources;
using ::mediapipe::tasks::metadata::SetExternalFile;
using ::mediapipe::tasks::vision::hand_detector::proto::
    HandDetectorGraphOptions;
using ::mediapipe::tasks::vision::hand_landmarker::proto::
    HandLandmarkerGraphOptions;
using ::mediapipe::tasks::vision::hand_landmarker::proto::
    HandLandmarksDetectorGraphOptions;

constexpr char kImageTag[] = "IMAGE";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kHandRectNextFrameTag[] = "HAND_RECT_NEXT_FRAME";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kPalmDetectionsTag[] = "PALM_DETECTIONS";
constexpr char kPalmRectsTag[] = "PALM_RECTS";
constexpr char kPreviousLoopbackCalculatorName[] = "PreviousLoopbackCalculator";
constexpr char kHandDetectorTFLiteName[] = "hand_detector.tflite";
constexpr char kHandLandmarksDetectorTFLiteName[] =
    "hand_landmarks_detector.tflite";

struct HandLandmarkerOutputs {
  Source<std::vector<NormalizedLandmarkList>> landmark_lists;
  Source<std::vector<LandmarkList>> world_landmark_lists;
  Source<std::vector<NormalizedRect>> hand_rects_next_frame;
  Source<std::vector<ClassificationList>> handednesses;
  Source<std::vector<NormalizedRect>> palm_rects;
  Source<std::vector<Detection>> palm_detections;
  Source<Image> image;
};

// Sets the base options in the sub tasks.
absl::Status SetSubTaskBaseOptions(const ModelAssetBundleResources& resources,
                                   HandLandmarkerGraphOptions* options,
                                   bool is_copy) {
  ASSIGN_OR_RETURN(const auto hand_detector_file,
                   resources.GetModelFile(kHandDetectorTFLiteName));
  SetExternalFile(hand_detector_file,
                  options->mutable_hand_detector_graph_options()
                      ->mutable_base_options()
                      ->mutable_model_asset(),
                  is_copy);
  ASSIGN_OR_RETURN(const auto hand_landmarks_detector_file,
                   resources.GetModelFile(kHandLandmarksDetectorTFLiteName));
  SetExternalFile(hand_landmarks_detector_file,
                  options->mutable_hand_landmarks_detector_graph_options()
                      ->mutable_base_options()
                      ->mutable_model_asset(),
                  is_copy);
  return absl::OkStatus();
}

}  // namespace

// A "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph" performs hand
// landmarks detection. The HandLandmarkerGraph consists of two subgraphs:
// HandDetectorGraph and MultipleHandLandmarksDetectorGraph.
// MultipleHandLandmarksDetectorGraph detects landmarks from bounding boxes
// produced by HandDetectorGraph. HandLandmarkerGraph tracks the landmarks over
// time, and skips the HandDetectorGraph. If the tracking is lost or the detectd
// hands are less than configured max number hands, HandDetectorGraph would be
// triggered to detect hands.
//
// Accepts CPU input images and outputs Landmarks on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform hand landmarks detection on.
//
// Outputs:
//   LANDMARKS: - std::vector<NormalizedLandmarkList>
//     Vector of detected hand landmarks.
//   WORLD_LANDMARKS - std::vector<LandmarkList>
//     Vector of detected hand landmarks in world coordinates.
//   HAND_RECT_NEXT_FRAME - std::vector<NormalizedRect>
//     Vector of the predicted rects enclosing the same hand RoI for landmark
//     detection on the next frame.
//   HANDEDNESS - std::vector<ClassificationList>
//     Vector of classification of handedness.
//   PALM_RECTS - std::vector<NormalizedRect>
//     Detected palm bounding boxes in normalized coordinates.
//   PALM_DETECTIONS - std::vector<Detection>
//     Detected palms with maximum `num_hands` specified in options.
//   IMAGE - Image
//     The input image that the hand landmarker runs on and has the pixel data
//     stored on the target storage (CPU vs GPU).
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph"
//   input_stream: "IMAGE:image_in"
//   output_stream: "LANDMARKS:hand_landmarks"
//   output_stream: "WORLD_LANDMARKS:world_hand_landmarks"
//   output_stream: "HAND_RECT_NEXT_FRAME:hand_rect_next_frame"
//   output_stream: "HANDEDNESS:handedness"
//   output_stream: "PALM_RECTS:palm_rects"
//   output_stream: "PALM_DETECTIONS:palm_detections"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.hand_landmarker.proto.HandLandmarkerGraphOptions.ext] {
//       base_options {
//          model_asset {
//            file_name: "hand_landmarker.task"
//          }
//       }
//       hand_detector_graph_options {
//         base_options {
//            model_asset {
//              file_name: "palm_detection.tflite"
//            }
//         }
//         min_detection_confidence: 0.5
//         num_hands: 2
//       }
//       hand_landmarks_detector_graph_options {
//         base_options {
//              model_asset {
//                file_name: "hand_landmark_lite.tflite"
//              }
//           }
//           min_detection_confidence: 0.5
//       }
//     }
//   }
// }
class HandLandmarkerGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    if (sc->Options<HandLandmarkerGraphOptions>()
            .base_options()
            .has_model_asset()) {
      ASSIGN_OR_RETURN(
          const auto* model_asset_bundle_resources,
          CreateModelAssetBundleResources<HandLandmarkerGraphOptions>(sc));
      // Copies the file content instead of passing the pointer of file in
      // memory if the subgraph model resource service is not available.
      MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          *model_asset_bundle_resources,
          sc->MutableOptions<HandLandmarkerGraphOptions>(),
          !sc->Service(::mediapipe::tasks::core::kModelResourcesCacheService)
               .IsAvailable()));
    }
    ASSIGN_OR_RETURN(
        auto hand_landmarker_outputs,
        BuildHandLandmarkerGraph(sc->Options<HandLandmarkerGraphOptions>(),
                                 graph[Input<Image>(kImageTag)], graph));
    hand_landmarker_outputs.landmark_lists >>
        graph[Output<std::vector<NormalizedLandmarkList>>(kLandmarksTag)];
    hand_landmarker_outputs.world_landmark_lists >>
        graph[Output<std::vector<LandmarkList>>(kWorldLandmarksTag)];
    hand_landmarker_outputs.hand_rects_next_frame >>
        graph[Output<std::vector<NormalizedRect>>(kHandRectNextFrameTag)];
    hand_landmarker_outputs.handednesses >>
        graph[Output<std::vector<ClassificationList>>(kHandednessTag)];
    hand_landmarker_outputs.palm_rects >>
        graph[Output<std::vector<NormalizedRect>>(kPalmRectsTag)];
    hand_landmarker_outputs.palm_detections >>
        graph[Output<std::vector<Detection>>(kPalmDetectionsTag)];
    hand_landmarker_outputs.image >> graph[Output<Image>(kImageTag)];

    // TODO remove when support is fixed.
    // As mediapipe GraphBuilder currently doesn't support configuring
    // InputStreamInfo, modifying the CalculatorGraphConfig proto directly.
    CalculatorGraphConfig config = graph.GetConfig();
    for (int i = 0; i < config.node_size(); ++i) {
      if (config.node(i).calculator() == kPreviousLoopbackCalculatorName) {
        auto* info = config.mutable_node(i)->add_input_stream_info();
        info->set_tag_index("LOOP");
        info->set_back_edge(true);
        break;
      }
    }
    return config;
  }

 private:
  // Adds a mediapipe hand landmark detection graph into the provided
  // builder::Graph instance.
  //
  // tasks_options: the mediapipe tasks module HandLandmarkerGraphOptions.
  // image_in: (mediapipe::Image) stream to run hand landmark detection on.
  // graph: the mediapipe graph instance to be updated.
  absl::StatusOr<HandLandmarkerOutputs> BuildHandLandmarkerGraph(
      const HandLandmarkerGraphOptions& tasks_options, Source<Image> image_in,
      Graph& graph) {
    const int max_num_hands =
        tasks_options.hand_detector_graph_options().num_hands();

    auto& previous_loopback = graph.AddNode(kPreviousLoopbackCalculatorName);
    image_in >> previous_loopback.In("MAIN");
    auto prev_hand_rects_from_landmarks =
        previous_loopback[Output<std::vector<NormalizedRect>>("PREV_LOOP")];

    auto& min_size_node =
        graph.AddNode("NormalizedRectVectorHasMinSizeCalculator");
    prev_hand_rects_from_landmarks >> min_size_node.In("ITERABLE");
    min_size_node.GetOptions<CollectionHasMinSizeCalculatorOptions>()
        .set_min_size(max_num_hands);
    auto has_enough_hands = min_size_node.Out("").Cast<bool>();

    auto image_for_hand_detector =
        DisallowIf(image_in, has_enough_hands, graph);

    auto& hand_detector =
        graph.AddNode("mediapipe.tasks.vision.hand_detector.HandDetectorGraph");
    hand_detector.GetOptions<HandDetectorGraphOptions>().CopyFrom(
        tasks_options.hand_detector_graph_options());
    image_for_hand_detector >> hand_detector.In("IMAGE");
    auto hand_rects_from_hand_detector = hand_detector.Out("HAND_RECTS");

    auto& hand_association = graph.AddNode("HandAssociationCalculator");
    hand_association.GetOptions<HandAssociationCalculatorOptions>()
        .set_min_similarity_threshold(tasks_options.min_tracking_confidence());
    prev_hand_rects_from_landmarks >>
        hand_association[Input<std::vector<NormalizedRect>>::Multiple("")][0];
    hand_rects_from_hand_detector >>
        hand_association[Input<std::vector<NormalizedRect>>::Multiple("")][1];
    auto hand_rects = hand_association.Out("");

    auto& clip_hand_rects =
        graph.AddNode("ClipNormalizedRectVectorSizeCalculator");
    clip_hand_rects.GetOptions<ClipVectorSizeCalculatorOptions>()
        .set_max_vec_size(max_num_hands);
    hand_rects >> clip_hand_rects.In("");
    auto clipped_hand_rects = clip_hand_rects.Out("");

    auto& hand_landmarks_detector_graph = graph.AddNode(
        "mediapipe.tasks.vision.hand_landmarker."
        "MultipleHandLandmarksDetectorGraph");
    hand_landmarks_detector_graph
        .GetOptions<HandLandmarksDetectorGraphOptions>()
        .CopyFrom(tasks_options.hand_landmarks_detector_graph_options());
    image_in >> hand_landmarks_detector_graph.In("IMAGE");
    clipped_hand_rects >> hand_landmarks_detector_graph.In("HAND_RECT");

    auto landmarks = hand_landmarks_detector_graph.Out(kLandmarksTag);
    auto world_landmarks =
        hand_landmarks_detector_graph.Out(kWorldLandmarksTag);
    auto hand_rects_for_next_frame =
        hand_landmarks_detector_graph.Out(kHandRectNextFrameTag);
    auto handedness = hand_landmarks_detector_graph.Out(kHandednessTag);

    auto& image_property = graph.AddNode("ImagePropertiesCalculator");
    image_in >> image_property.In("IMAGE");
    auto image_size = image_property.Out("SIZE");

    auto& deduplicate = graph.AddNode("HandLandmarksDeduplicationCalculator");
    landmarks >> deduplicate.In("MULTI_LANDMARKS");
    world_landmarks >> deduplicate.In("MULTI_WORLD_LANDMARKS");
    hand_rects_for_next_frame >> deduplicate.In("MULTI_ROIS");
    handedness >> deduplicate.In("MULTI_CLASSIFICATIONS");
    image_size >> deduplicate.In("IMAGE_SIZE");

    auto filtered_landmarks =
        deduplicate[Output<std::vector<NormalizedLandmarkList>>(
            "MULTI_LANDMARKS")];
    auto filtered_world_landmarks =
        deduplicate[Output<std::vector<LandmarkList>>("MULTI_WORLD_LANDMARKS")];
    auto filtered_hand_rects_for_next_frame =
        deduplicate[Output<std::vector<NormalizedRect>>("MULTI_ROIS")];
    auto filtered_handedness =
        deduplicate[Output<std::vector<ClassificationList>>(
            "MULTI_CLASSIFICATIONS")];

    // Back edge.
    filtered_hand_rects_for_next_frame >> previous_loopback.In("LOOP");

    // TODO: Replace PassThroughCalculator with a calculator that
    // converts the pixel data to be stored on the target storage (CPU vs GPU).
    auto& pass_through = graph.AddNode("PassThroughCalculator");
    image_in >> pass_through.In("");

    return {{
        /* landmark_lists= */ filtered_landmarks,
        /* world_landmark_lists= */ filtered_world_landmarks,
        /* hand_rects_next_frame= */ filtered_hand_rects_for_next_frame,
        /* handedness= */ filtered_handedness,
        /* palm_rects= */
        hand_detector[Output<std::vector<NormalizedRect>>(kPalmRectsTag)],
        /* palm_detections */
        hand_detector[Output<std::vector<Detection>>(kPalmDetectionsTag)],
        /* image */
        pass_through[Output<Image>("")],
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerGraph);

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
