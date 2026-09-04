# ESP32 MNIST TensorFlow Lite Micro Demo

End-to-end TinyML pipeline:

MNIST -> train tiny CNN -> full INT8 quantization -> .tflite -> C++ model header -> ESP32 inference -> Serial output.

## 1. Laptop setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## 2. Train and export

From the project root:

```bash
python train_and_export.py
```

This downloads MNIST through Keras, trains a small CNN, evaluates it, performs full integer (INT8) quantization, and generates:

- `model/mnist_int8.tflite`
- `esp32_mnist/main/model_data.cc`
- `esp32_mnist/main/mnist_test_data.h`

The last file contains 10 test images so the ESP32 can run inference without a camera.

## 3. ESP-IDF setup

Install ESP-IDF for your ESP32 board. The project uses Espressif's `esp-tflite-micro` component.

The official Espressif component can be added with:

```bash
cd esp32_mnist
idf.py add-dependency "esp-tflite-micro"
```

Then select the classic ESP32 target:

```bash
idf.py set-target esp32
```

If your board is ESP32-S3, use:

```bash
idf.py set-target esp32s3
```

The code itself is written against TensorFlow Lite Micro APIs and does not use a camera.

## 4. Build and flash

Connect the ESP32 by USB and find its serial port.

Linux:
```bash
idf.py -p /dev/ttyUSB0 build flash monitor
```

Windows example:
```powershell
idf.py -p COM5 build flash monitor
```

The exact port depends on your machine.

## 5. Expected serial output

You should see something similar to:

```text
ESP32 MNIST INT8 demo
Model loaded
Input: 28 x 28 x 1
Input scale: ...
Input zero point: ...
Running 10 test images...

Test 0 | expected=7 | predicted=7 | confidence=0.98 | time=...
...
```

The exact accuracy, confidence, and inference time depend on the trained model, ESP32 variant, CPU frequency, and build configuration.

## 6. What this experiment demonstrates

- Training on a standard public dataset
- Model conversion to TensorFlow Lite
- Full INT8 quantization
- Embedded model packaging as a C/C++ array
- TensorFlow Lite Micro inference
- On-device input quantization
- On-device output dequantization
- Inference timing
- Serial benchmarking

This is deliberately a camera-free first deployment test. Once it works, the next step is replacing the embedded MNIST test image with camera input.
