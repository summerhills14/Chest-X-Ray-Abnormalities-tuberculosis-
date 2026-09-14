"""
Command-line single-image inference, mirroring predict_xray() from the
original notebook.

Usage:
    python predict.py path/to/xray.png
    python predict.py path/to/xray.png --model models/best_tb_cnn.keras --threshold 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)


def preprocess(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("L").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr[None, ..., None]


def main():
    parser = argparse.ArgumentParser(description="Predict TB vs Normal from a chest X-ray image.")
    parser.add_argument("image", type=str, help="Path to the chest X-ray image (png/jpg).")
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).parent / "models" / "best_tb_cnn.keras"),
        help="Path to a .keras model file.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for TB.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    arr = preprocess(args.image)
    prob_tb = float(model.predict(arr, verbose=0)[0][0])
    prediction = "Tuberculosis" if prob_tb >= args.threshold else "Normal"

    print(f"Image        : {args.image}")
    print(f"Model        : {args.model}")
    print(f"TB probability: {prob_tb:.4f}")
    print(f"Prediction    : {prediction}")
    print("\nNote: research prototype only, not a validated diagnostic tool.")


if __name__ == "__main__":
    main()
