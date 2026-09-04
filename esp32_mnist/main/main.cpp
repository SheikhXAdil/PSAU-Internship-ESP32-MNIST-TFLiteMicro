#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <inttypes.h>

#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "mnist_test_data.h"

extern const unsigned char g_mnist_model_data[];
extern const unsigned int g_mnist_model_data_len;

namespace {

constexpr size_t kTensorArenaSize = 100 * 1024;

alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

int8_t quantize_input(uint8_t pixel, const TfLiteQuantizationParams& q) {
    const float real_value = static_cast<float>(pixel) / 255.0f;

    const int32_t quantized =
        static_cast<int32_t>(std::lround(real_value / q.scale))
        + q.zero_point;

    const int32_t clamped =
        std::max<int32_t>(
            -128,
            std::min<int32_t>(127, quantized)
        );

    return static_cast<int8_t>(clamped);
}

float dequantize_output(int8_t value, const TfLiteQuantizationParams& q) {
    return (static_cast<float>(value) - static_cast<float>(q.zero_point)) *
           q.scale;
}

}  // namespace

extern "C" void app_main(void) {
    printf("\n");
    printf("========================================\n");
    printf(" ESP32 MNIST INT8 TensorFlow Lite Micro\n");
    printf("========================================\n");

    // Map the flatbuffer model into a TFLM model object.
    model = tflite::GetModel(g_mnist_model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: model schema version mismatch\n");
        printf("Model: %" PRIu32 ", TFLM: %" PRIu32 "\n",
       model->version(),
       static_cast<uint32_t>(TFLITE_SCHEMA_VERSION));
        return;
    }

    // Register only operations used by our CNN.
    // Conv2D and FullyConnected include their fused activations.
    static tflite::MicroMutableOpResolver<8> resolver;

    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddShape() != kTfLiteOk ||
        resolver.AddPack() != kTfLiteOk ||
        resolver.AddStridedSlice() != kTfLiteOk) {
        printf("ERROR: failed to register TFLM operators\n");
        return;
    }

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors() failed\n");
        printf("Try increasing kTensorArenaSize.\n");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    printf("Model loaded successfully\n");
    printf("Input type: %d\n", static_cast<int>(input->type));
    printf("Input dims: ");

    for (int i = 0; i < input->dims->size; ++i) {
        printf("%d", input->dims->data[i]);
        if (i + 1 < input->dims->size) {
            printf(" x ");
        }
    }
    printf("\n");

    printf("Input quantization: scale=%f zero_point=%" PRId32 "\n",
       input->params.scale,
       input->params.zero_point);

    printf("Output quantization: scale=%f zero_point=%" PRId32 "\n",
       output->params.scale,
       output->params.zero_point);

    printf("Free heap after model allocation: %lu bytes\n",
           static_cast<unsigned long>(heap_caps_get_free_size(MALLOC_CAP_8BIT)));

    printf("\nRunning %d MNIST test images...\n\n", MNIST_TEST_COUNT);

    for (int test = 0; test < MNIST_TEST_COUNT; ++test) {
        // Copy and quantize one 28x28 image into the TFLM input tensor.
        for (int i = 0; i < MNIST_IMAGE_SIZE; ++i) {
            input->data.int8[i] =
                static_cast<int8_t>(
                    quantize_input(mnist_images[test][i], input->params)
                );
        }

        const int64_t start_us = esp_timer_get_time();

        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Test %d: ERROR - Invoke() failed\n", test);
            continue;
        }

        const int64_t elapsed_us = esp_timer_get_time() - start_us;

        int predicted = 0;
        float best_score = -1.0f;

        for (int c = 0; c < 10; ++c) {
            const float score =
                dequantize_output(output->data.int8[c], output->params);

            if (score > best_score) {
                best_score = score;
                predicted = c;
            }
        }

        const int expected = mnist_labels[test];

        printf(
            "Test %02d | expected=%d | predicted=%d | confidence=%.3f | "
            "time=%lld us (%lld ms) | %s\n",
            test,
            expected,
            predicted,
            best_score,
            static_cast<long long>(elapsed_us),
            static_cast<long long>(elapsed_us / 1000),
            predicted == expected ? "OK" : "WRONG"
        );

        // Give FreeRTOS a chance to run other tasks.
        vTaskDelay(pdMS_TO_TICKS(250));
    }

    printf("\nFinished.\n");
    printf("The model is executing entirely on the ESP32.\n");

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
