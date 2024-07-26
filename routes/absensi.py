import io
import cv2
import logging
import sqlite3
import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from fastapi.responses import StreamingResponse
from keras_facenet import FaceNet
from utils.db_setup import get_db
from utils.auth import get_current_user
from models.Mahasiswa import Mahasiswa
from models.Absensi import Absensi

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

facenet = FaceNet()
model = tf.keras.models.load_model("./utils/FaceNet/model_FRHIMA_Gray_FaceNet.h5")
label_encoder = np.load(
    "./utils/FaceNet/labels_encoder_FaceNet.npy", allow_pickle=True
).item()

face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")

router = APIRouter()


def apply_face_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )
    if len(faces) == 0:
        raise ValueError("No face detected")
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_image = gray[y : y + h, x : x + w]
    return face_image


def detect_and_crop_face(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]
    face = image[y : y + h, x : x + w]

    return face


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image


@router.post("/predict")
async def predict(
    file: UploadFile = File(...), db: sqlite3.Connection = Depends(get_db)
):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))

        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        cropped_face = detect_and_crop_face(open_cv_image)

        cropped_face_pil = Image.fromarray(
            cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        )

        processed_image = preprocess_image(cropped_face_pil)
        logger.debug(f"Processed image shape: {processed_image.shape}")
        logger.debug(f"Processed image data: {processed_image}")

        # Generate FaceNet embeddings
        embeddings = facenet.embeddings(processed_image)

        # Predict the class
        prediction = model.predict(embeddings)
        predicted_class_index = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
        probability = float(prediction[0][predicted_class_index])

        logger.debug(f"Predicted class: {predicted_class}, Probability: {probability}")

        if probability is None or predicted_class is None:
            raise HTTPException(
                status_code=404, detail="No face detected or recognized"
            )

        cursor = db.cursor()
        cursor.execute(
            "SELECT NIM, Nama, Divisi FROM mahasiswa WHERE NIM = ?", (predicted_class,)
        )
        row = cursor.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Mahasiswa tidak ditemukan")

        return Mahasiswa(NIM=row[0], Nama=row[1], Divisi=row[2])
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def read_absensi(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT id_absensi, id_mahasiswa, m.nama, DATE(tanggal_absensi), TIME(jam_absensi), keterangan
        FROM absensi a
        JOIN mahasiswa m ON a.id_mahasiswa = m.nim
        """
    )
    rows = cursor.fetchall()

    absensi_list = [
        {
            "id_absensi": row[0],
            "id_mahasiswa": row[1],
            "nama": row[2],
            "tanggal_absensi": row[3],
            "jam_absensi": row[4],
            "keterangan": row[5],
        }
        for row in rows
    ]

    return absensi_list


@router.get("/export", dependencies=[Depends(get_current_user)])
async def export_as_excel(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT id_absensi, id_mahasiswa, m.nama, m.divisi, DATE(tanggal_absensi), TIME(jam_absensi), keterangan
        FROM absensi a
        JOIN mahasiswa m ON a.id_mahasiswa = m.nim
        """
    )
    rows = cursor.fetchall()

    absensi_list = [
        {
            "id_absensi": row[0],
            "id_mahasiswa": row[1],
            "nama": row[2],
            "divisi": row[3],
            "tanggal_absensi": row[4],
            "jam_absensi": row[5],
            "keterangan": row[6],
        }
        for row in rows
    ]

    df = pd.DataFrame(absensi_list)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Absensi")

    output.seek(0)

    response = StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response.headers["Content-Disposition"] = "attachment; filename=absensi.xlsx"

    return response


@router.post("/", response_model=Absensi, dependencies=[Depends(get_current_user)])
async def create_absensi(
    absensi: Absensi,
    db: sqlite3.Connection = Depends(get_db),
):
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO absensi (id_mahasiswa, tanggal_absensi, jam_absensi, keterangan) VALUES (?, ?, ?, ?)",
        (
            absensi.id_mahasiswa,
            absensi.tanggal_absensi,
            absensi.jam_absensi,
            absensi.keterangan,
        ),
    )

    db.commit()
    absensi.id_absensi = cursor.lastrowid

    return absensi


@router.get("/leaderboard")
async def leaderboards(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT COUNT(a.keterangan) AS leaderboards, m.Nama
        FROM absensi a
        INNER JOIN mahasiswa m ON m.NIM = a.id_mahasiswa
        GROUP BY m.Nama
        ORDER BY leaderboards DESC
        """
    )

    rows = cursor.fetchall()

    leaderboard = [{"total": row[0], "nama": row[1]} for row in rows]

    return leaderboard


# @app.get("/api/absensi/{nim}")
# async def read_absensi_by_id(nim: str, db: sqlite3.Connection = Depends(get_db)):
#     cursor = db.cursor()
#     cursor.execute(
#         """
#         SELECT id_absensi, id_mahasiswa, m.nama, DATE(tanggal_absensi), TIME(jam_absensi), keterangan
#         FROM absensi a
#         JOIN mahasiswa m ON a.id_mahasiswa = m.nim
#         WHERE id_mahasiswa = ?
#         """,
#         (nim,),
#     )
#     row = cursor.fetchone()

#     if row is None:
#         raise HTTPException(status_code=404, detail="Mahasiswa not found")

#     return {
#         "id_absensi": row[0],
#         "id_mahasiswa": row[1],
#         "nama": row[2],
#         "tanggal_absensi": row[3],
#         "jam_absensi": row[4],
#         "keterangan": row[5],
#     }
