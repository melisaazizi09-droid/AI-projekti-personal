
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


# ============================================================
# SETTINGS
# ============================================================

MODEL_FOLDER = "model"
MODEL_FILE = "keras_model.h5"
LABELS_FILE = "labels.txt"

IMAGE_SIZE = (224, 224)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / MODEL_FOLDER / MODEL_FILE
LABELS_PATH = BASE_DIR / MODEL_FOLDER / LABELS_FILE


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():
    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            label = line.strip()

            if not label:
                continue

            parts = label.split(" ", 1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(label)

    return labels


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image_path):

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    image = Image.open(image_path).convert("RGB")

    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data[0] = normalized_image_array

    return data


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(MODEL_PATH, compile=False)

class_names = load_labels()


# ============================================================
# SPORT PREDICTION
# ============================================================

def predict_sport(image_path):

    data = prepare_image(image_path)

    prediction = model.predict(
        data,
        verbose=0
    )

    index = int(np.argmax(prediction[0]))

    sport_class = class_names[index]

    confidence = float(prediction[0][index])

    return {
        "sport": sport_class,
        "confidence": confidence
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    image_path = BASE_DIR / "test.jpg"

    result = predict_sport(image_path)

    print("Detected sport:", result["sport"])

    print(
        "Confidence:",
        f'{result["confidence"] * 100:.2f}%'
    )
