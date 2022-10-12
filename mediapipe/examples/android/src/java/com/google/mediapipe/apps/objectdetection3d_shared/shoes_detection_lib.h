#ifndef SHOES_DETECTION_LIBRARY_H
#define SHOES_DETECTION_LIBRARY_H

#include <cstdlib>
#include <memory>
#include <string>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"

struct KeypointWithHidden{
  std::int32_t keypoint_id;
  cv::Point3f PointCoordinates3D;
  cv::Point2f PointCoordinates2D;
  bool hidden;
};

struct ShoesBoundingBox{
  std::int32_t Id;
  std::vector<KeypointWithHidden> Keypoints;
  cv::Mat Transformation;
};

class MPShoesDetector {
public:
  MPShoesDetector(
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
    const int imageWidth = 0,
    const int imageHeight = 0);

  virtual void DetectShoes(
    const cv::Mat &camera_frame,
    int* numObjects,
    ShoesBoundingBox* detected_objects,
    const std::chrono::milliseconds& timestamp);

private:
  absl::Status InitShoesDetector(
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
    const int imageHeight);

  absl::Status DetectShoesWithStatus(
    const cv::Mat &camera_frame,
    int* numObjects,
    ShoesBoundingBox* detected_objects,
    const std::chrono::milliseconds& timestamp);

  static constexpr auto kInputStream = "input_image";
  static constexpr auto kOutputStream_detected_objects = "detected_objects";
  static constexpr auto kOutputStream_landmarks_empty = "landmarks_empty";


  static const std::string graphConfig;

  mediapipe::CalculatorGraph graph;
  mediapipe::GlCalculatorHelper gpu_helper;

  std::unique_ptr<mediapipe::OutputStreamPoller> detected_objects_poller_ptr;
  std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_empty_poller_ptr;
};

#ifdef __cplusplus
extern "C" {
#endif

MPShoesDetector*
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
  const int imageHeight);

void MPShoesDetectorDestruct(MPShoesDetector *detector);

void MPShoesDetectorDetectShoes(
  MPShoesDetector *detector,
  const cv::Mat &camera_frame,
  int* numObjects,
  ShoesBoundingBox* detected_objects,
  const std::chrono::milliseconds& timestamp);

#ifdef __cplusplus
};
#endif
#endif