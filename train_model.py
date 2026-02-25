import os
import cv2 as cv
import numpy as np
import pickle

def train_model():
    os.makedirs("trainer", exist_ok=True)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    DIR = "dataset"
    if not os.path.exists(DIR):
        return "Dataset folder not found."

    people = os.listdir(DIR)  # folders named "Name_StudentID"
    people = [p for p in people if os.path.isdir(os.path.join(DIR, p))]

    if not people:
        return "No registered faces found."

    features = []
    labels = []

    for idx, person in enumerate(people):
        path = os.path.join(DIR, person)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            resized = cv.resize(gray, (200, 200))
            features.append(resized)
            labels.append(idx)

    if not features:
        return "No images found to train."

    face_recognizer.train(features, np.array(labels))
    face_recognizer.save("trainer/face_trained.yml")

    with open("trainer/people.pkl", "wb") as f:
        pickle.dump(people, f)

    print("Training complete")
    return f"Training complete. {len(people)} person(s) trained."