import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("utils/model_FRHIMA_RGB04.h5")


def apply_lbp(image):
    face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]
    face_image = image[y : y + h, x : x + w]

    return face_image


def recognize_face(image: np.array):
    # lbp_image = apply_lbp(image)
    try:
        cropped_face = apply_lbp(image)
    except ValueError as e:
        print(e)
        return None

    # Preprocess the image as per your model's requirements
    face_image = cv2.resize(cropped_face, (150, 150))
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=0)

    prediction = model.predict(face_image)
    print(prediction)
    label_index = np.argmax(prediction, axis=1)[0]

    labels = [
        "2105101038",
        "2105101039",
        "2105101068",
        "2105101071",
        "2105101079",
        "2205101047",
        "2305101033",
        "2305101072",
        "2305101083",
        "2305101118",
        "2305101121",
        "2305101130",
        "2305101131",
    ]
    label = labels[label_index]

    return label


image_path = "public/2305101118/2305101118_5.jpg"
image = cv2.imread(image_path)

print(image)

label = recognize_face(image)

print("Label: ", label)
