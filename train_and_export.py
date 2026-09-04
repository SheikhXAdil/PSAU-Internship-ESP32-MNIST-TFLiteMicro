import os
import pathlib
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
ESP_MAIN = ROOT / "esp32_mnist" / "main"
MODEL_DIR.mkdir(exist_ok=True)
ESP_MAIN.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load standard MNIST dataset
# ---------------------------------------------------------------------
print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Convert to float32 [0, 1].
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# ---------------------------------------------------------------------
# 2. Tiny CNN
# ---------------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print("\nTraining...")
model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128,
    verbose=2,
)

print("\nFloat32 model evaluation:")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss:     {loss:.5f}")
print(f"Test accuracy: {accuracy:.4f}")

# Save a normal Keras model for reference.
model.save(MODEL_DIR / "mnist_float.keras")

# ---------------------------------------------------------------------
# 3. Full integer (INT8) TFLite quantization
# ---------------------------------------------------------------------
def representative_dataset():
    # A few hundred representative samples are enough for this demo.
    for image in x_train[:300]:
        yield [image[np.newaxis, ...].astype(np.float32)]

print("\nConverting to fully INT8 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

tflite_path = MODEL_DIR / "mnist_int8.tflite"
tflite_path.write_bytes(tflite_model)

print(f"TFLite model: {tflite_path}")
print(f"TFLite size:  {len(tflite_model):,} bytes")

# ---------------------------------------------------------------------
# 4. Verify the quantized model on the laptop before deployment
# ---------------------------------------------------------------------
print("\nChecking INT8 model on laptop...")
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Input details:")
print(input_details)
print("Output details:")
print(output_details)

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

correct = 0
for i in range(1000):
    image = x_test[i:i+1]

    q = np.round(image / input_scale + input_zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details["index"], q)
    interpreter.invoke()

    output_q = interpreter.get_tensor(output_details["index"])[0]
    output = (output_q.astype(np.float32) - output_zero_point) * output_scale

    predicted = int(np.argmax(output))
    correct += predicted == int(y_test[i])

print(f"INT8 accuracy on first 1000 test images: {correct / 1000:.4f}")

# ---------------------------------------------------------------------
# 5. Generate C++ model_data.cc
# ---------------------------------------------------------------------
def write_c_array(path, symbol, data, bytes_per_line=12):
    with open(path, "w", encoding="utf-8") as f:
        f.write("#include <cstdint>\n\n")
        f.write(f"alignas(8) extern const unsigned char {symbol}[] = {{\n")

        for i in range(0, len(data), bytes_per_line):
            chunk = data[i:i + bytes_per_line]
            f.write("    " + ", ".join(f"0x{b:02x}" for b in chunk))
            if i + bytes_per_line < len(data):
                f.write(",")
            f.write("\n")

        f.write("};\n")
        f.write(f"const unsigned int {symbol}_len = {len(data)};\n")

model_cc = ESP_MAIN / "model_data.cc"
write_c_array(model_cc, "g_mnist_model_data", tflite_model)

# ---------------------------------------------------------------------
# 6. Generate a few MNIST test images for the ESP32
# ---------------------------------------------------------------------
# Store raw uint8 pixels (0..255), not normalized floats.
num_device_tests = 10
device_images = np.round(x_test[:num_device_tests, :, :, 0] * 255.0).astype(np.uint8)
device_labels = y_test[:num_device_tests].astype(np.uint8)

with open(ESP_MAIN / "mnist_test_data.h", "w", encoding="utf-8") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define MNIST_TEST_COUNT {num_device_tests}\n")
    f.write("#define MNIST_IMAGE_SIZE 784\n\n")

    f.write("static const uint8_t mnist_labels[MNIST_TEST_COUNT] = {\n    ")
    f.write(", ".join(str(int(x)) for x in device_labels))
    f.write("\n};\n\n")

    f.write("static const uint8_t mnist_images[MNIST_TEST_COUNT][MNIST_IMAGE_SIZE] = {\n")
    for image in device_images:
        f.write("    {\n")
        flat = image.flatten()
        for i in range(0, len(flat), 28):
            row = flat[i:i+28]
            f.write("        " + ", ".join(str(int(x)) for x in row) + ",\n")
        f.write("    },\n")
    f.write("};\n")

print(f"Generated: {model_cc}")
print(f"Generated: {ESP_MAIN / 'mnist_test_data.h'}")
print("\nDone. Your ESP32 project is now ready to build.")
