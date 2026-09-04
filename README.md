# ESP32 MNIST — TensorFlow Lite Micro

An end-to-end **TinyML deployment experiment** for running a quantized neural network on an **ESP32 microcontroller** using **TensorFlow Lite Micro (TFLM)** and **ESP-IDF**.

This project was developed during my internship at the **Research & Development Center, Prince Sattam bin Abdulaziz University (PSAU)** as an experimental part of my broader work on **Edge AI deployment for resource-constrained hardware**.

The project demonstrates the complete pipeline from training a small neural network to deploying the model and performing inference directly on an ESP32.

## Overview

The project uses the **MNIST handwritten digit dataset** to train a small convolutional neural network (CNN), convert the trained model to TensorFlow Lite, perform **full INT8 quantization**, embed the model into the ESP32 firmware, and run inference using TensorFlow Lite Micro.

The complete workflow is:

```text
MNIST Dataset
      ↓
Train Small CNN
      ↓
Evaluate Model
      ↓
Full INT8 Quantization
      ↓
TensorFlow Lite Model
      ↓
Convert Model to C/C++ Data
      ↓
ESP32 + TensorFlow Lite Micro
      ↓
On-Device Inference
      ↓
Serial Output & Timing
```

The deployment is intentionally **camera-free**. A small set of MNIST test images is embedded into the firmware so the ML deployment pipeline can first be validated independently of camera and sensor integration.

## Features

* MNIST dataset training
* Lightweight CNN model
* Full integer **INT8 quantization**
* TensorFlow Lite model conversion
* Embedded C/C++ model representation
* TensorFlow Lite Micro inference
* ESP32 deployment using ESP-IDF
* Input quantization
* Output dequantization
* Embedded MNIST test images
* On-device inference timing
* Serial-based benchmarking

## Repository Structure

```text
PSAU-Internship-ESP32-MNIST-TFLiteMicro/
│
├── esp32_mnist/
│   └── ESP-IDF application
│       ├── main/
│       │   ├── main.cpp
│       │   ├── model_data.cc
│       │   ├── model_data.h
│       │   └── mnist_test_data.h
│       │
│       ├── CMakeLists.txt
│       └── idf_component.yml
│
├── model/
│   └── Generated TensorFlow Lite model
│
├── requirements.txt
├── train_and_export.py
└── README.md
```

### ESP32 Application

The `esp32_mnist/main/` directory contains the files required by the ESP-IDF application:

* `main.cpp` — ESP32 application and inference logic
* `model_data.cc` — embedded TensorFlow Lite model data
* `model_data.h` — model data interface
* `mnist_test_data.h` — embedded MNIST test images

The generated ESP-IDF `build/` directory is intentionally **not included** in the repository because it contains build artifacts that can be regenerated locally.

## Requirements

### Python

* Python
* TensorFlow / Keras
* NumPy
* Dependencies listed in `requirements.txt`

### ESP32

* ESP32 development board
* ESP-IDF
* TensorFlow Lite Micro
* USB connection for flashing and serial monitoring

The ESP-IDF application uses Espressif's `esp-tflite-micro` component for TensorFlow Lite Micro integration.

## 1. Python Environment

From the project root, create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## 2. Train and Export the Model

From the project root:

```bash
python train_and_export.py
```

The script:

1. Downloads the MNIST dataset through Keras.
2. Trains a small CNN.
3. Evaluates the model.
4. Performs full integer INT8 quantization.
5. Exports the TensorFlow Lite model.
6. Generates the C/C++ representation of the model.
7. Generates the MNIST test data used by the ESP32 application.

The generated artifacts include:

```text
model/mnist_int8.tflite
esp32_mnist/main/model_data.cc
esp32_mnist/main/mnist_test_data.h
```

The test-data header contains a small set of MNIST images, allowing the ESP32 to perform inference without requiring a camera.

## 3. ESP-IDF Setup

Install and configure **ESP-IDF** for your ESP32 development board.

Move into the ESP-IDF application:

```bash
cd esp32_mnist
```

The project declares its ESP-IDF component dependency through:

```text
idf_component.yml
```

If the TensorFlow Lite Micro component needs to be added manually, it can be installed with:

```bash
idf.py add-dependency "esp-tflite-micro"
```

Set the target for a classic ESP32:

```bash
idf.py set-target esp32
```

For an ESP32-S3:

```bash
idf.py set-target esp32s3
```

The application uses TensorFlow Lite Micro APIs and does not require a camera.

## 4. Build and Flash

Connect the ESP32 to the development machine through USB.

### Linux

```bash
idf.py -p /dev/ttyUSB0 build flash monitor
```

### Windows

```bash
idf.py -p COM5 build flash monitor
```

Replace the serial port with the one corresponding to your ESP32.

The `build/` directory will be generated automatically by ESP-IDF when the project is built.

## 5. Expected Serial Output

After flashing the firmware, the ESP32 runs inference against the embedded MNIST test images.

The output follows a format similar to:

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

The exact predictions, confidence values, and inference times depend on the trained model and ESP32 configuration.

## Technical Pipeline

### Model Training

A lightweight CNN is trained using the MNIST dataset. The model is intentionally kept small to make it suitable for deployment on a resource-constrained microcontroller.

### INT8 Quantization

The trained model is converted using **full integer INT8 quantization**.

Quantization reduces the numerical representation used by the model and makes it more suitable for constrained embedded inference.

### Model Packaging

The resulting TensorFlow Lite model is converted into C/C++ data so it can be compiled directly into the ESP32 firmware.

```text
mnist_int8.tflite
        ↓
model_data.cc / model_data.h
        ↓
ESP32 firmware
```

### TensorFlow Lite Micro Inference

The embedded model is loaded and executed directly on the ESP32 using TensorFlow Lite Micro.

The application handles:

* Model loading
* Tensor allocation
* Input preparation
* Input quantization
* Model invocation
* Output dequantization
* Prediction extraction
* Confidence reporting
* Inference timing

## What This Experiment Demonstrates

This experiment provides practical experience with the complete TinyML deployment workflow:

* Training a model using a public dataset
* Converting a trained model to TensorFlow Lite
* Performing full INT8 quantization
* Packaging an ML model for embedded deployment
* Using TensorFlow Lite Micro
* Developing with ESP-IDF
* Running ML inference directly on an ESP32
* Handling quantized input and output tensors
* Measuring on-device inference time
* Benchmarking embedded inference

The camera-free design deliberately isolates the ML deployment process from camera and sensor integration.

## Why MNIST?

MNIST was used as a controlled first deployment workload rather than as the final application.

The objective was to validate the fundamental process of taking a trained ML model and successfully running it on a microcontroller:

```text
Training
   ↓
Conversion
   ↓
Quantization
   ↓
Model Packaging
   ↓
Embedded Runtime
   ↓
On-Device Inference
```

Using a small and well-understood dataset made it possible to focus on the **deployment pipeline and embedded constraints** before moving toward more computationally demanding computer-vision workloads.

## Internship Context

This project was developed as part of my internship at the **Research & Development Center, Prince Sattam bin Abdulaziz University (PSAU)**.

The broader internship focused on investigating and developing a practical **Edge AI deployment pipeline for resource-constrained hardware**.

The ESP32 experiment served as an initial TinyML deployment proof of concept. The internship later progressed toward more complex computer-vision workloads, including:

* **YOLOv7 drone detection on AMB82-Mini**
* **YOLOv7 drone detection on Raspberry Pi**
* **YOLO11n drone detection and Raspberry Pi optimization**
* Model conversion and optimization
* Quantization
* On-device performance evaluation
* Investigation of hardware and runtime constraints

These experiments formed part of a broader investigation into the path from **model training to deployment and real-world inference on edge hardware**.

## Related Internship Work

This project is part of the larger internship archive:

**[PSAU Internship — Edge AI Deployment Pipeline](https://github.com/SheikhXAdil/PSAU-Internship-Archive)**

The archive contains the broader internship documentation, presentation, experiments, and implementations.

The individual projects are maintained in separate repositories to make each implementation easier to explore independently.

## Documentation

The complete internship research, methodology, deployment findings, and broader analysis are available in the main internship archive.

**[PSAU Internship Archive](https://github.com/SheikhXAdil/PSAU-Internship-Archive)**

The archive contains:

* `Edge AI Deployment Pipeline Documentation.pdf`
* `Edge AI Deployment Pipeline Presentation.pptx`

These documents provide the broader context of the Edge AI research and the different hardware and deployment experiments carried out during the internship.

## Outcome

The project successfully demonstrated an end-to-end **TinyML deployment pipeline from model training to on-device inference on an ESP32**.

The experiment provided practical exposure to the constraints and considerations involved in deploying machine-learning models on microcontrollers, including:

* Model size
* Quantization
* Embedded model packaging
* Runtime compatibility
* Memory constraints
* On-device inference
* Inference timing

It also provided a foundation for the subsequent Edge AI work involving object detection models on more capable edge platforms.

## Status

**Completed experimental project.**

This repository is preserved as part of my **PSAU internship work** and documents an end-to-end TinyML deployment experiment using **ESP32, TensorFlow Lite Micro, ESP-IDF, and INT8 quantization**.

It serves as the ESP32 component of the broader **PSAU Edge AI deployment work** and complements the more advanced drone-detection experiments maintained in separate repositories.
