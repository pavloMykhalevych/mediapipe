#include "shoes_detection_lib.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

MPShoesDetector::MPShoesDetector(
  const char *boxLandmarkModelPath,
  const char *oidModelPath,
  const char *allowedLabels,
  const int maxMumObjects,
  const bool usePrevLandmarks,
  const double minTrackingThreshold,
  const double minDetectingThreshold,
  const double inputFocalX,
  const double inputFocalY,
  const double inputPrincipalPointX,
  const double inputPrincipalPointY,
  const int imageWidth,
  const int imageHeight) {
  const auto status = InitShoesDetector(
    boxLandmarkModelPath,
    oidModelPath,
    allowedLabels,
    maxMumObjects,
    usePrevLandmarks,
    minTrackingThreshold,
    minDetectingThreshold,
    inputFocalX,
    inputFocalY,
    inputPrincipalPointX,
    inputPrincipalPointY,
    imageWidth,
    imageHeight);
  if (!status.ok()) {
    LOG(INFO) << "Failed constructing MPShoesDetector.";
    LOG(INFO) << status.message();
  }
}

void MPShoesDetector::DetectShoes(
  const cv::Mat &camera_frame,
  int* numObjects,
  ShoesBoundingBox* detected_objects,
  const std::chrono::milliseconds& timestamp)
{
  const auto status = DetectShoesWithStatus(camera_frame, numObjects, detected_objects, timestamp);
  if (!status.ok()) {
    LOG(INFO) << "MPShoesDetector::DetectShoes failed: " << status.message();
  }
}

absl::Status
MPShoesDetector::InitShoesDetector(
  const char *boxLandmarkModelPath,
  const char *oidModelPath,
  const char *allowedLabels,
  const int maxMumObjects,
  const bool usePrevLandmarks,
  const double minTrackingThreshold,
  const double minDetectingThreshold,
  const double inputFocalX,
  const double inputFocalY,
  const double inputPrincipalPointX,
  const double inputPrincipalPointY,
  const int imageWidth,
  const int imageHeight)
{
  auto inputFocalX_new = inputFocalX ;
  auto inputFocalY_new = inputFocalY;
  auto inputPrincipalPointX_new = inputPrincipalPointX;
  auto inputPrincipalPointY_new = inputPrincipalPointY;
  if (imageWidth != 0){
    int half_width = imageWidth / 2.0;
    int half_height = imageHeight / 2.0;
    inputFocalX_new = inputFocalX / half_width;
    inputFocalY_new = inputFocalY / half_height;
    inputPrincipalPointX_new = - (inputPrincipalPointX - half_width) / half_width;
    inputPrincipalPointY_new = - (inputPrincipalPointY - half_height) / half_height;
  }
  // Prepare graph config.
  auto preparedGraphConfig = absl::StrReplaceAll(graphConfig, {{"$boxLandmarkModelPath", boxLandmarkModelPath}});
  preparedGraphConfig = absl::StrReplaceAll(preparedGraphConfig, {{"$oidModelPath", oidModelPath}});
  preparedGraphConfig = absl::StrReplaceAll( preparedGraphConfig, { {"$allowedLabels", allowedLabels} });
  preparedGraphConfig = absl::StrReplaceAll( preparedGraphConfig, { {"$maxMumObjects", std::to_string(maxMumObjects)} });
  preparedGraphConfig = absl::StrReplaceAll( preparedGraphConfig, { {"$usePrevLandmarks", usePrevLandmarks ? "true" : "false"} });
  preparedGraphConfig = absl::StrReplaceAll( preparedGraphConfig, { {"$minTrackingThreshold", std::to_string(minTrackingThreshold)} });
  preparedGraphConfig = absl::StrReplaceAll( preparedGraphConfig, { {"$minDetectingThreshold", std::to_string(minDetectingThreshold)} });
  preparedGraphConfig = absl::StrReplaceAll( preparedGraphConfig, { {"$inputFocalX", std::to_string(inputFocalX_new)} });
  preparedGraphConfig = absl::StrReplaceAll( preparedGraphConfig, { {"$inputFocalY", std::to_string(inputFocalY_new)} });
  preparedGraphConfig = absl::StrReplaceAll( preparedGraphConfig, { {"$inputPrincipalPointX", std::to_string(inputPrincipalPointX_new)} });
  preparedGraphConfig = absl::StrReplaceAll( preparedGraphConfig, { {"$inputPrincipalPointY", std::to_string(inputPrincipalPointY_new)} });

  LOG(INFO) << "Get calculator graph config contents: " << preparedGraphConfig;

  mediapipe::CalculatorGraphConfig config =
    mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(preparedGraphConfig);
  LOG(INFO) << "Initialize the calculator graph.";

  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());
  
  LOG(INFO) << "Start running the calculator graph.";

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller detected_objects_poller,
                   graph.AddOutputStreamPoller(kOutputStream_detected_objects));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarks_empty_poller,
                   graph.AddOutputStreamPoller(kOutputStream_landmarks_empty));

  detected_objects_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
      std::move(detected_objects_poller));
  landmarks_empty_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
      std::move(landmarks_empty_poller));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "MPShoesDetector constructed successfully.";

  return absl::OkStatus();
}

absl::Status
MPShoesDetector::DetectShoesWithStatus(
  const cv::Mat &camera_frame,
  int* numObjects,
  ShoesBoundingBox* detected_objects,
  const std::chrono::milliseconds& timestamp)
{
  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
    mediapipe::ImageFormat::SRGBA,
    camera_frame.cols,
    camera_frame.rows,
    mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  
  camera_frame.copyTo(input_frame_mat);
  // Send image packet into the graph.
  MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &timestamp, this]() -> absl::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(timestamp.count()))));
          return absl::OkStatus();
        }));
  
  mediapipe::Packet is_landmarks_empty_packet;
  if (!landmarks_empty_poller_ptr ||
      !landmarks_empty_poller_ptr->Next(
          &is_landmarks_empty_packet)) {
    return absl::CancelledError(
        "Failed during getting next frame_annotation_packet.");
  }

  auto &is_landmarks_empty =
      is_landmarks_empty_packet
          .Get<bool>();
  
  if(is_landmarks_empty){
    numObjects = 0;
    LOG(INFO) << "Return detections empty.";
    return absl::OkStatus();
  }
  
  // Get frame annotation.
  mediapipe::Packet frame_annotation_packet;
  if (!detected_objects_poller_ptr ||
      !detected_objects_poller_ptr->Next(
          &frame_annotation_packet)) {
    return absl::CancelledError(
        "Failed during getting next frame_annotation_packet.");
  }
  
  LOG(INFO) << "Get frame annotation.";
  auto &frame_annotation =
      frame_annotation_packet
          .Get<::mediapipe::FrameAnnotation>();
  LOG(INFO) << "Got frame annotation.";
  int objects_number = 0;
  for (const auto& annotation : frame_annotation.annotations()) {
    std::vector<KeypointWithHidden> keypoints;
    for (const auto& keypoint : annotation.keypoints()) {
      keypoints.push_back(KeypointWithHidden{
        keypoint.id(),
        cv::Point3f(keypoint.point_3d().x(), keypoint.point_3d().y(), keypoint.point_3d().z()),
        cv::Point2f(keypoint.point_2d().x() * camera_frame.cols, keypoint.point_2d().y() * camera_frame.rows),
        keypoint.hidden()});
      LOG(INFO) << "keypoints[" <<  keypoints.size() << "] = " << keypoints.back().PointCoordinates2D;
    }
    const int objectId = annotation.object_id();
    std::vector<float> rotation;
    std::vector<float> translation;
    for (const auto& rotation_param : annotation.rotation()){
      rotation.push_back(rotation_param);
    }
    for (const auto& translation_param : annotation.translation()){
      translation.push_back(translation_param);
    }
    cv::Mat transformation = cv::Mat::eye(4, 4, CV_32FC1);

    cv::Mat rotationMatrix = cv::Mat(3, 3, CV_8U, rotation.data());
    rotationMatrix.copyTo(transformation(cv::Rect(0, 0, 3, 3)));

    transformation.at<float>(0, 3) = translation[0];
    transformation.at<float>(1, 3) = translation[1];
    transformation.at<float>(2, 3) = translation[2];
    LOG(INFO) << "transformation = " << transformation;

    detected_objects[objects_number] = ShoesBoundingBox{annotation.object_id(), keypoints, transformation};
    ++objects_number;
  }
  *numObjects = objects_number;
  LOG(INFO) << "numObjects = " << *numObjects;
  

  return absl::OkStatus();
}


#ifdef __cplusplus
extern "C" {
#endif

MPShoesDetector *
MPShoesDetectorConstruct(
  const char *boxLandmarkModelPath,
  const char *oidModelPath,
  const char *allowedLabels,
  const int maxMumObjects,
  const bool usePrevLandmarks,
  const double minTrackingThreshold,
  const double minDetectingThreshold,
  const double inputFocalX,
  const double inputFocalY,
  const double inputPrincipalPointX,
  const double inputPrincipalPointY,
  const int imageWidth,
  const int imageHeight)
{
  return new MPShoesDetector(
    boxLandmarkModelPath,
    oidModelPath,
    allowedLabels,
    maxMumObjects,
    usePrevLandmarks,
    minTrackingThreshold,
    minDetectingThreshold,
    inputFocalX,
    inputFocalY,
    inputPrincipalPointX,
    inputPrincipalPointY,
    imageWidth,
    imageHeight);
}

void MPShoesDetectorDestruct(MPShoesDetector *detector) {
  delete detector;
}

void MPShoesDetectorDetectShoes(
  MPShoesDetector *detector,
  const cv::Mat &camera_frame,
  int* numObjects,
  ShoesBoundingBox* detected_objects,
  const std::chrono::milliseconds& timestamp)
{
  LOG(INFO) << "MPShoesDetector DetectShoes.";
  detector->DetectShoes(camera_frame, numObjects, detected_objects, timestamp);
  LOG(INFO) << "MPShoesDetector DetectShoes finished.";
}

#ifdef __cplusplus
};
#endif

const std::string MPShoesDetector::graphConfig = R"pb(
# MediaPipe graph that performs shoes pose detection on CPU.
# Input image. (GPUBuffer)
input_stream: "input_image"
# Collection of detected 3D objects, represented as a FrameAnnotation.
output_stream: "detected_objects"

output_stream: "landmarks_empty"

# Defines side packets for further use in the graph.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:0:box_landmark_model_path"
  output_side_packet: "PACKET:1:allowed_labels"
  output_side_packet: "PACKET:2:max_num_objects"
  output_side_packet: "PACKET:3:use_prev_landmarks"
  output_side_packet: "PACKET:4:min_tracking_threshold"
  output_side_packet: "PACKET:5:min_detecting_threshold"
  output_side_packet: "PACKET:6:input_focal_x"
  output_side_packet: "PACKET:7:input_focal_y"
  output_side_packet: "PACKET:8:input_principal_point_x"
  output_side_packet: "PACKET:9:input_principal_point_y"
  output_side_packet: "PACKET:10:oid_model_path"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { string_value: "$boxLandmarkModelPath" }
      packet { string_value: "$allowedLabels" }
      packet { int_value: $maxMumObjects }
      packet { bool_value: $usePrevLandmarks }
      packet { double_value: $minTrackingThreshold }
      packet { double_value: $minDetectingThreshold }
      packet { double_value: $inputFocalX }
      packet { double_value: $inputFocalY }
      packet { double_value: $inputPrincipalPointX }
      packet { double_value: $inputPrincipalPointY }
      packet { string_value: "$oidModelPath"  }
    }
  }
}

# Loads the file in the specified path into a blob.
node {
  calculator: "LocalFileContentsCalculator"
  input_side_packet: "FILE_PATH:0:oid_model_path"
  input_side_packet: "FILE_PATH:1:box_landmark_model_path"
  output_side_packet: "CONTENTS:0:oid_model_blob"
  output_side_packet: "CONTENTS:1:box_landmark_model_blob"
}

# Converts the input blob into a TF Lite model.
node {
  calculator: "TfLiteModelCalculator"
  input_side_packet: "MODEL_BLOB:box_landmark_model_blob"
  output_side_packet: "MODEL:box_landmark_model"
}

# Converts the input blob into a TF Lite model.
node {
  calculator: "TfLiteModelCalculator"
  input_side_packet: "MODEL_BLOB:oid_model_blob"
  output_side_packet: "MODEL:oid_model"
}

# Subgraph that detects faces and corresponding landmarks.
node {
  calculator: "ObjectronGpuExtendedSubgraph"
  input_stream: "IMAGE_GPU:input_image"
  input_side_packet: "OID_MODEL:oid_model"
  input_side_packet: "BOX_MODEL:box_landmark_model"
  input_side_packet: "LABELS_CSV:allowed_labels"
  input_side_packet: "MAX_NUM_OBJECTS:max_num_objects"
  input_side_packet: "USE_PREV_LANDMARKS:use_prev_landmarks"
  input_side_packet: "MIN_TRACKING_THRESHOLD:min_tracking_threshold"
  input_side_packet: "MIN_DETECTING_THRESHOLD:min_detecting_threshold"
  input_side_packet: "INPUT_FOCAL_X:input_focal_x"
  input_side_packet: "INPUT_FOCAL_Y:input_focal_y"
  input_side_packet: "INPUT_PRINCIPAL_POINT_X:input_principal_point_x"
  input_side_packet: "INPUT_PRINCIPAL_POINT_Y:input_principal_point_y"
  output_stream: "FRAME_ANNOTATION:detected_objects"
  output_stream: "EMPTY:landmarks_empty"
}
)pb";