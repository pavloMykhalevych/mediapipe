// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/calculators/util/check_vector_empty_calculator.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

typedef CheckVectorEmptyCalculator<
    std::vector<::mediapipe::NormalizedLandmarkList>>
    CheckLandmarksVectorEmptyCalculator;
REGISTER_CALCULATOR(CheckLandmarksVectorEmptyCalculator);

typedef CheckVectorEmptyCalculator<
    std::vector<::mediapipe::Detection>>
    CheckDetectionsVectorEmptyCalculator;
REGISTER_CALCULATOR(CheckDetectionsVectorEmptyCalculator);

typedef CheckVectorEmptyCalculator<
    std::vector<::mediapipe::NormalizedRect>>
    CheckRectsVectorEmptyCalculator;
REGISTER_CALCULATOR(CheckRectsVectorEmptyCalculator);

} // namespace mediapipe
