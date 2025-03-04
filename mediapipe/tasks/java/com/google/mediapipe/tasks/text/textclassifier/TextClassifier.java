// Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

package com.google.mediapipe.tasks.text.textclassifier;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.tasks.components.container.proto.ClassificationsProto;
import com.google.mediapipe.tasks.components.processors.ClassifierOptions;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.text.textclassifier.proto.TextClassifierGraphOptionsProto;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Performs classification on text.
 *
 * <p>This API expects a TFLite model with (optional) <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata</a> that contains
 * the mandatory (described below) input tensors, output tensor, and the optional (but recommended)
 * label items as AssociatedFiles with type TENSOR_AXIS_LABELS per output classification tensor.
 *
 * <p>Metadata is required for models with int32 input tensors because it contains the input process
 * unit for the model's Tokenizer. No metadata is required for models with string input tensors.
 *
 * <ul>
 *   <li>Input tensors
 *       <ul>
 *         <li>Three input tensors ({@code kTfLiteInt32}) of shape {@code [batch_size x
 *             bert_max_seq_len]} representing the input ids, mask ids, and segment ids. This input
 *             signature requires a Bert Tokenizer process unit in the model metadata.
 *         <li>Or one input tensor ({@code kTfLiteInt32}) of shape {@code [batch_size x
 *             max_seq_len]} representing the input ids. This input signature requires a Regex
 *             Tokenizer process unit in the model metadata.
 *         <li>Or one input tensor ({@code kTfLiteString}) that is shapeless or has shape {@code
 *             [1]} containing the input string.
 *       </ul>
 *   <li>At least one output tensor ({@code kTfLiteFloat32}/{@code kBool}) with:
 *       <ul>
 *         <li>{@code N} classes and shape {@code [1 x N]}
 *         <li>optional (but recommended) label map(s) as AssociatedFile-s with type
 *             TENSOR_AXIS_LABELS, containing one label per line. The first such AssociatedFile (if
 *             any) is used to fill the {@code class_name} field of the results. The {@code
 *             display_name} field is filled from the AssociatedFile (if any) whose locale matches
 *             the {@code display_names_locale} field of the {@code TextClassifierOptions} used at
 *             creation time ("en" by default, i.e. English). If none of these are available, only
 *             the {@code index} field of the results will be filled.
 *       </ul>
 * </ul>
 */
public final class TextClassifier implements AutoCloseable {
  private static final String TAG = TextClassifier.class.getSimpleName();
  private static final String TEXT_IN_STREAM_NAME = "text_in";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("TEXT:" + TEXT_IN_STREAM_NAME));

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("CLASSIFICATION_RESULT:classification_result_out"));

  private static final int CLASSIFICATION_RESULT_OUT_STREAM_INDEX = 0;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.text.text_classifier.TextClassifierGraph";
  private final TaskRunner runner;

  static {
    System.loadLibrary("mediapipe_tasks_text_jni");
    ProtoUtil.registerTypeName(
        ClassificationsProto.ClassificationResult.class,
        "mediapipe.tasks.components.containers.proto.ClassificationResult");
  }

  /**
   * Creates a {@link TextClassifier} instance from a model file and the default {@link
   * TextClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the text model with metadata in the assets.
   * @throws MediaPipeException if there is is an error during {@link TextClassifier} creation.
   */
  public static TextClassifier createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, TextClassifierOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link TextClassifier} instance from a model file and the default {@link
   * TextClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the text model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link TextClassifier} creation.
   */
  public static TextClassifier createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, TextClassifierOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link TextClassifier} instance from {@link TextClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param options a {@link TextClassifierOptions} instance.
   * @throws MediaPipeException if there is an error during {@link TextClassifier} creation.
   */
  public static TextClassifier createFromOptions(Context context, TextClassifierOptions options) {
    OutputHandler<TextClassificationResult, Void> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<TextClassificationResult, Void>() {
          @Override
          public TextClassificationResult convertToTaskResult(List<Packet> packets) {
            try {
              return TextClassificationResult.create(
                  PacketGetter.getProto(
                      packets.get(CLASSIFICATION_RESULT_OUT_STREAM_INDEX),
                      ClassificationsProto.ClassificationResult.getDefaultInstance()),
                  packets.get(CLASSIFICATION_RESULT_OUT_STREAM_INDEX).getTimestamp());
            } catch (InvalidProtocolBufferException e) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.INTERNAL.ordinal(), e.getMessage());
            }
          }

          @Override
          public Void convertToTaskInput(List<Packet> packets) {
            return null;
          }
        });
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<TextClassifierOptions>builder()
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(options)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    return new TextClassifier(runner);
  }

  /**
   * Constructor to initialize a {@link TextClassifier} from a {@link TaskRunner}.
   *
   * @param runner a {@link TaskRunner}.
   */
  private TextClassifier(TaskRunner runner) {
    this.runner = runner;
  }

  /**
   * Performs classification on the input text.
   *
   * @param inputText a {@link String} for processing.
   */
  public TextClassificationResult classify(String inputText) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(TEXT_IN_STREAM_NAME, runner.getPacketCreator().createString(inputText));
    return (TextClassificationResult) runner.process(inputPackets);
  }

  /** Closes and cleans up the {@link TextClassifier}. */
  @Override
  public void close() {
    runner.close();
  }

  /** Options for setting up a {@link TextClassifier}. */
  @AutoValue
  public abstract static class TextClassifierOptions extends TaskOptions {

    /** Builder for {@link TextClassifierOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the text classifier task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the optional {@link ClassifierOptions} controling classification behavior, such as
       * score threshold, number of results, etc.
       */
      public abstract Builder setClassifierOptions(ClassifierOptions classifierOptions);

      public abstract TextClassifierOptions build();
    }

    abstract BaseOptions baseOptions();

    abstract Optional<ClassifierOptions> classifierOptions();

    public static Builder builder() {
      return new AutoValue_TextClassifier_TextClassifierOptions.Builder();
    }

    /** Converts a {@link TextClassifierOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
      baseOptionsBuilder.mergeFrom(convertBaseOptionsToProto(baseOptions()));
      TextClassifierGraphOptionsProto.TextClassifierGraphOptions.Builder taskOptionsBuilder =
          TextClassifierGraphOptionsProto.TextClassifierGraphOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder);
      if (classifierOptions().isPresent()) {
        taskOptionsBuilder.setClassifierOptions(classifierOptions().get().convertToProto());
      }
      return CalculatorOptions.newBuilder()
          .setExtension(
              TextClassifierGraphOptionsProto.TextClassifierGraphOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }
}
