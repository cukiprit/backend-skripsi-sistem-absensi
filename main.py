import sqlite3
import cv2
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sklearn.decomposition import PCA
from io import BytesIO
from typing import List
from utils.db_setup import get_db, create_table
from models.Mahasiswa import Mahasiswa
from models.Absensi import Absensi

origins = ["http://localhost", "http://localhost:5173"]

app = FastAPI()
create_table()
get_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("utils/model fix 1/model_FRHIMA_GRAY19.h5")
labels = np.load("utils/model fix 1/labels1.npy")


def apply_face_detection(image):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    if len(faces) == 0:
        raise ValueError("No face detected")
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_image = gray[y : y + h, x : x + w]
    return face_image


def recognize_face(image: np.array):
    try:
        cropped_face = apply_face_detection(image)
    except ValueError as e:
        print(e)
        return None

    face_image = cv2.resize(cropped_face, (130, 130))  # Update size to 130x130
    face_image = np.repeat(
        face_image[:, :, np.newaxis], 3, axis=2
    )  # Convert to 3 channels
    face_image = face_image / 255.0  # Normalize
    face_image = np.expand_dims(face_image, axis=0)  # Expand dimensions for batch

    print(f"Preprocessed face image shape: {face_image.shape}")
    print(f"Preprocessed face image values: {face_image[0, :, :, 0]}")

    prediction = model.predict(face_image)

    print(f"Model raw output: {prediction}")

    label_index = np.argmax(prediction, axis=1)[0]

    return labels[label_index]


@app.post("/api/recognize-face", response_model=Mahasiswa)
async def recognize_face_endpoint(
    file: UploadFile = File(...),
    db: sqlite3.Connection = Depends(get_db),
):
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, np.uint8)

    if image.size == 0:
        raise HTTPException(status_code=400, detail="Invalid image data")

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=404, detail="Failed to decode image")

    label = recognize_face(image)

    if label is None:
        raise HTTPException(status_code=404, detail="No face detected or recognized")

    cursor = db.cursor()
    cursor.execute("SELECT NIM, Nama, Divisi FROM mahasiswa WHERE NIM = ?", (label,))
    row = cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Mahasiswa tidak ditemukan")

    return Mahasiswa(NIM=row[0], Nama=row[1], Divisi=row[2])


@app.post("/api/process-image")
async def process_image_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    label, cropped_face = recognize_face(image)

    if cropped_face is None:
        raise HTTPException(status_code=400, detail="No face detected")

    # Resize to 150x150
    face_image = cv2.resize(cropped_face, (130, 130))

    # Prepare the image for response in the same format (JPG)
    _, buffer = cv2.imencode(".jpg", face_image)
    byte_io = BytesIO(buffer)
    return StreamingResponse(byte_io, media_type="image/jpeg")


@app.get("/api/mahasiswa", response_model=List[Mahasiswa])
def read_mahasiswa(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT NIM, Nama, Divisi FROM mahasiswa")
    rows = cursor.fetchall()
    return [Mahasiswa(NIM=row[0], Nama=row[1], Divisi=row[2]) for row in rows]


@app.get("/api/mahasiswa/{nim}", response_model=Mahasiswa)
def read_mahasiswa_by_id(nim: str, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT NIM, Nama, Divisi FROM mahasiswa WHERE NIM = ?", (nim,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Mahasiswa not found")
    return Mahasiswa(NIM=row[0], Nama=row[1], Divisi=row[2])


@app.post("/api/mahasiswa", response_model=Mahasiswa)
def create_mahasiswa(mahasiswa: Mahasiswa, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO (NIM, Nama, Divisi) VALUES ('?', '?', '?')",
            (mahasiswa.NIM, mahasiswa.Nama, mahasiswa.Divisi),
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="NIM already exists")
    return mahasiswa


@app.get("/api/absensi", response_model=List[Absensi])
def read_absensi(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute(
        "SELECT id_absensi, id_mahasiswa, tanggal_absensi, jam_absensi, keterangan FROM absensi"
    )
    rows = cursor.fetchall()
    return [
        Absensi(
            id_absensi=row[0],
            id_mahasiswa=row[1],
            tanggal_absensi=row[2],
            jam_absensi=row[3],
            keterangan=row[4],
        )
        for row in rows
    ]


@app.post("/api/absensi", response_model=Absensi)
def create_absensi(absensi: Absensi, db: sqlite3.Connection = Depends(get_db)):
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
