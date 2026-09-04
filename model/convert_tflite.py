from pathlib import Path
import re

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path("mnist_int8.tflite")
OUTPUT_PATH = Path("mnist_model.h")

ARRAY_NAME = "mnist_model"


# ============================================================
# Convert .tflite -> C header
# ============================================================

def convert_tflite_to_header():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.absolute()}"
        )

    model_data = MODEL_PATH.read_bytes()

    print(f"Input model : {MODEL_PATH}")
    print(f"Model size  : {len(model_data):,} bytes")

    with open(OUTPUT_PATH, "w") as f:

        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("// TensorFlow Lite model\n")
        f.write(f"// Original size: {len(model_data):,} bytes\n\n")

        f.write(f"alignas(8) const unsigned char {ARRAY_NAME}[] = {{\n")

        for i in range(0, len(model_data), 12):

            chunk = model_data[i:i + 12]

            values = ", ".join(
                f"0x{byte:02x}"
                for byte in chunk
            )

            f.write(f"    {values},\n")

        f.write("};\n\n")

        f.write(
            f"const unsigned int {ARRAY_NAME}_len "
            f"= {len(model_data)};\n"
        )

    print(f"Output header: {OUTPUT_PATH}")
    print("Conversion complete.")


if __name__ == "__main__":
    convert_tflite_to_header()