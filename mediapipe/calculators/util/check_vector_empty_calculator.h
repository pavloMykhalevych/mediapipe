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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_COUNTING_VECTOR_SIZE_CALCULATOR_H
#define MEDIAPIPE_CALCULATORS_UTIL_COUNTING_VECTOR_SIZE_CALCULATOR_H

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"

namespace mediapipe {

// A calculator that counts the size of the input vector. It was created to
// aid in polling packets in the output stream synchronously. If there is
// a clock stream, it will output a value of 0 even if the input vector stream
// is empty. If not, it will output some value only if there is an input vector.
// The clock stream must have the same time stamp as the vector stream, and
// it must be the stream where packets are transmitted while the graph is
// running. (e.g. Any input stream of graph)
//
// It is designed to be used like:
//
// node {
//   calculator: "CheckVectorEmptyCalculator"
//   input_stream: "VECTOR:input_vector"
//   output_stream: "EMPTY:vector_empty"
// }

template <typename VectorT>
class CheckVectorEmptyCalculator : public CalculatorBase {
public:
  static ::mediapipe::Status GetContract(CalculatorContract *cc) {
   
    RET_CHECK(cc->Inputs().HasTag("VECTOR"));
    cc->Inputs().Tag("VECTOR").Set<VectorT>();
    RET_CHECK(cc->Outputs().HasTag("EMPTY"));
    cc->Outputs().Tag("EMPTY").Set<bool>();

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext *cc) {
    std::unique_ptr<bool> vector_empty;
    bool test;
    if (!cc->Inputs().Tag("VECTOR").IsEmpty()) {
      const auto &landmarks = cc->Inputs().Tag("VECTOR").Get<VectorT>();
      vector_empty = absl::make_unique<bool>(landmarks.empty());
      test = landmarks.empty();
    } else {
      vector_empty = absl::make_unique<bool>(true);
      test = true;
    }
    LOG(INFO) << "CheckVectorEmptyCalculator vector empty - " << test;
    cc->Outputs().Tag("EMPTY").Add(vector_empty.release(), cc->InputTimestamp());

    return ::mediapipe::OkStatus();
  };
};

} // namespace mediapipe

#endif // MEDIAPIPE_CALCULATORS_UTIL_COUNTING_VECTOR_SIZE_CALCULATOR_H
