#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <stdint.h>
#include <time.h>
#define INPUT_SIZE (240 * 320 * 3)
#define OUTPUT_SIZE (4420 * 2)

namespace {
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<30>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddExp());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLog());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTanh());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSquaredDifference());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRsqrt());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSub());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTransposeConv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCos());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSin());

  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddResizeNearestNeighbor());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddBatchMatMul());
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 3000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(g_model), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
  interpreter.input(0)->data.f[0] = 1.f;
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(g_model);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 3000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Check if the predicted output is within a small range of the
  // expected output
  float epsilon = 0.05f;
  constexpr int kNumTestValues = 4;
  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};

  for (int i = 0; i < kNumTestValues; ++i) {
    interpreter.input(0)->data.f[0] = golden_inputs[i];
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    float y_pred = interpreter.output(0)->data.f[0];
    TFLITE_CHECK_LE(abs(sin(golden_inputs[i]) - y_pred), epsilon);
  }

  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_model);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 5000*1024;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  int8_t input_data0[INPUT_SIZE];
  //int64_t input_data1[1];
  //input_data1[0]=1;
  int8_t output_data[OUTPUT_SIZE];
  for(int i=0;i<INPUT_SIZE;i++){
    input_data0[i]=1;
  }


  TfLiteTensor* input0 = interpreter.input(0);
  TFLITE_CHECK_NE(input0, nullptr);

  // TfLiteTensor* input1 = interpreter.input(1);
  // TFLITE_CHECK_NE(input1, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  // float output_scale = output->params.scale;
  // int output_zero_point = output->params.zero_point;

  int8_t* inTensorData0 = tflite::GetTensorData<int8_t>(input0);
  memcpy(inTensorData0, input_data0, input0->bytes);
  // int64_t* inTensorData1 = tflite::GetTensorData<int64_t>(input1);
  // memcpy(inTensorData1, input_data1, input1->bytes);
  clock_t start, end;
  double cpu_time_used;
  start = clock();
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Execution time of one invoke: %f seconds\n", cpu_time_used);

  // memcpy(&output_data[0], tflite::GetTensorData<int8_t>(output), output->bytes);
  // printf("output = %d\n", output_data[0]);

  printf("arena_used_bytes = %ld\n", interpreter.arena_used_bytes());
  return kTfLiteOk;


  // Check if the predicted output is within a small range of the
  // expected output
  //float epsilon = 0.05;

  // constexpr int kNumTestValues = 4;
  //float golden_inputs_float[kNumTestValues] = {0.77, 1.57, 2.3, 3.14};

  // The int8 values are calculated using the following formula
  // (golden_inputs_float[i] / input->params.scale + input->params.scale)
  //int8_t golden_inputs_int8[kNumTestValues] = {-96, -63, -34, 0};

  // for (int i = 0; i < kNumTestValues; ++i) {
  //   input->data.int8[0] = golden_inputs_int8[i];
  //   TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  //   float y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
  //   TFLITE_CHECK_LE(abs(sin(golden_inputs_float[i]) - y_pred), epsilon);
  // }

  //return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  //tflite::InitializeTarget();
  //TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  //TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());
  MicroPrintf("~~~ALL TESTS PASSEDbvcxbvcxbvcx~~~\n");
  return kTfLiteOk;
}
