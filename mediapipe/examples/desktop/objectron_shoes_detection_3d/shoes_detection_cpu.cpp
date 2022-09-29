#include "shoes_detection_lib.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

#include "absl/flags/parse.h"
#include <iostream>

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  cv::VideoCapture capture;
  capture.open("D:/repos/feets_videos/desktop_camera/feets.mp4");
  if (!capture.isOpened()) {
    return -1;
  }

  constexpr char kWindowName[] = "MediaPipe";
  cv::namedWindow(kWindowName, 1);
  cv::resizeWindow(kWindowName, cv::Size(1280, 720));

  LOG(INFO) << "VideoCapture initialized.";

  constexpr char boxLandmarkModelPath[] = "mediapipe/modules/objectron/object_detection_3d_sneakers.tflite";
  constexpr char allowedLabels[] = "Footwear";
  constexpr int maxMumObjects = 5;
  constexpr bool usePrevLandmarks = true;
  constexpr float minTrackingThreshold = 0.7;
  constexpr float minDetectingThreshold = 0.7;
  constexpr float inputFocalX = 1184.1984137201562;
  constexpr float inputFocalY = 1186.2353791810572;
  constexpr float inputPrincipalPointX = 359.23453794433027;
  constexpr float inputPrincipalPointY = 618.1427265759373;
  int imageWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int imageHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

  MPShoesDetector *shoesDetector = MPShoesDetectorConstruct(
    boxLandmarkModelPath,
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

  std::cout << "FaceMeshDetector constructed.";

  std::cout << "Start grabbing and processing frames.";

  std::vector<ShoesBoundingBox> objects;
  objects.resize(maxMumObjects);
  bool grab_frames = true;
  int frame_count = 0;
  while (grab_frames) {
    // Capture opencv camera.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      std::cout << "Ignore empty frames from camera.";
      break;
    }

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    int objectsCount = 0;
    const float fps = capture.get(cv::CAP_PROP_FPS);
    const auto timestamp = std::chrono::milliseconds( static_cast<int>(frame_count * (1000 / fps)) ) ;

    MPShoesDetectorDetectShoes(shoesDetector, camera_frame, &objectsCount, objects.data(), timestamp);
    std::cout << "objectsCount = " << objectsCount << "\n";
    if (objectsCount > 0) {
      for (int i = 0; i < objectsCount; ++i){
        std::cout << "Results:";
        std::cout << "Transformation:" << objects[i].Transformation << "\n";

        for (const auto& keypoint : objects[i].Keypoints) {
          if(!keypoint.hidden)
          {
            cv::circle(camera_frame_raw, keypoint.PointCoordinates2D, 10, cv::Scalar(0, 0, 255));
          }
        }
      }
    }

    const int pressed_key = cv::waitKey(5);
    if (pressed_key >= 0 && pressed_key != 255)
      grab_frames = false;
    auto copy_frame = camera_frame_raw.clone();
    cv::resize(copy_frame, copy_frame, cv::Size(1280, 720));
    cv::imshow(kWindowName, copy_frame);

    ++frame_count;
  }

  std::cout << "Shutting down.";

  MPShoesDetectorDestruct(shoesDetector);
}