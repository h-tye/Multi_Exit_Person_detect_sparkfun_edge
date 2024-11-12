#include "main_functions.h"
#include <math.h>
#include <iostream>

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup(int model_idx) {
  tflite::InitializeTarget();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model based on what exit we're at
  model = tflite::GetModel(g_person_detect_model_data_array[model_idx]);

  //output what section of model we're in
  std::cout << "In Section " << model_idx << std::endl;

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Initialize operations
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // Create the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensor memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get input
  input = interpreter->input(0);
}

float* softmax(TfLiteTensor* tensor) {
  if (tensor->type != kTfLiteFloat32) {
    TF_LITE_REPORT_ERROR(error_reporter, "Output tensor is not Float32");
  }

  float* data = tensor->data.f;  
  int num_elements = 1;
  for (int i = 0; i < tensor->dims->size; i++) {
    num_elements *= tensor->dims->data[i];
  }

  float* prob = new float[num_elements];  
  float sum = 0.0;
  for (int i = 0; i < num_elements; i++) {
    prob[i] = expf(data[i]);
    sum += prob[i];
  }
  for (int i = 0; i < num_elements; i++) {
    prob[i] /= sum;
  }

  return prob;
}

float entropyCalc(float* prob, int size) {
  float entropy = 0.0;
  for (int i = 0; i < size; i++) {
    if (prob[i] > 0) {
      entropy -= prob[i] * logf(prob[i]);
    }
  }
  return entropy;
}

void loop() {
  
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }

  int8_t* current_input = input->data.int8;


  setup(0);

  int size = 3; //length of array of models

  for (int i = 1; i<=size; i++) {

    // Set the current input to the model's input tensor
    memcpy(input->data.int8, current_input, input->bytes);

    if (kTfLiteOk != interpreter->Invoke()) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    }

    TfLiteTensor* output = interpreter->output(0);
    
    float* prob = softmax(output);
    int num_elements = output->dims->data[output->dims->size - 1];
    float entropy = entropyCalc(prob, num_elements);
    float threshold = 2;  // Define threshold logic here

    //if entropy is less than threshold or we have reached final exit, then exit
    if (entropy < threshold || i == size) {
      int8_t person_score = output->data.uint8[kPersonIndex];
      int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
      RespondToDetection(error_reporter, person_score, no_person_score);
      delete[] prob;
      break;
    } 

    //else, first make sure output is int_8 type, then pass output into input
    else {
      if (output->type == kTfLiteInt8) {
        int8_t* output_data = output->data.int8;  // Safe to access as int8 data
        current_input = output_data; //set input to previous output
        //set up next stage of model
        setup(i);
      } 
      else {
        TF_LITE_REPORT_ERROR(error_reporter, "Unexpected output tensor type");
        return;  
      }
    
    }

    delete[] prob;
  }
}
