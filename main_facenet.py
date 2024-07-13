import io
import cv2
import logging
import sqlite3
import tensorflow as tf
import numpy as np

from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from keras_facenet import FaceNet
from utils.db_setup import get_db
from models.Mahasiswa import Mahasiswa
from models.Absensi import Absensi

origins = ["http://localhost", "http://localhost:5173"]

app = FastAPI()
get_db()

facenet = FaceNet()
model = tf.keras.models.load_model("./utils/FaceNet/model_FRHIMA_Gray_FaceNet.h5")
label_encoder = np.load(
    "./utils/FaceNet/labels_encoder_FaceNet.npy", allow_pickle=True
).item()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")


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


@app.post("/api/predict")
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


@app.get("/api/mahasiswa")
async def read_mahasiswa(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT NIM, Nama, Divisi FROM mahasiswa")
    rows = cursor.fetchall()

    return [{"NIM": row[0], "Nama": row[1], "Divisi": row[2]} for row in rows]


@app.get("/api/mahasiswa/{nim}")
async def read_mahasiswa_by_nim(nim: str, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT NIM, Nama, Divisi FROM mahasiswa WHERE NIM = ?", (nim,))
    row = cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Mahasiswa not found")

    return {"NIM": row[0], "Nama": row[1], "Divisi": row[2]}


@app.post("/api/mahasiswa")
async def create_mahasiswa(
    NIM: str, Nama: str, Divisi: str, db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()

    try:
        cursor.execute(
            "INSERT INTO mahasiswa (NIM, Nama, Divisi) VALUES (?, ?, ?)",
            (NIM, Nama, Divisi),
        )

        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=404, detail="NIM already exists")

    return {"NIM": NIM, "Nama": Nama, "Divisi": Divisi}


@app.put("/api/mahasiswa/{nim}")
async def update_mahasiswa(
    Nama: str, Divisi: str, nim: str, db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    cursor.execute(
        "UPDATE mahasiswa SET Nama = ?, Divisi = ? WHERE NIM = ?",
        (Nama, Divisi, nim),
    )

    db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Mahasiswa not found")

    return {"Nama": Nama, "Divisi": Divisi, "nim": nim}


@app.delete("/api/mahasiswa/{nim}")
async def delete_mahasiswa(nim: str, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("DELETE FROM mahasiswa WHERE NIM = ?", (nim,))
    db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Mahasiswa not found")

    return {"detail": "Mahasiswa deleted successfully"}


@app.get("/api/absensi")
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


@app.get("/api/absensi/export")
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


@app.post("/api/absensi")
async def create_absensi(
    id_mahasiswa: str,
    tanggal_absensi: str,
    jam_absensi: str,
    keterangan: str,
    db: sqlite3.Connection = Depends(get_db),
):
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO absensi (id_mahasiswa, tanggal_absensi, jam_absensi, keterangan) VALUES (?, ?, ?, ?)",
        (
            id_mahasiswa,
            tanggal_absensi,
            jam_absensi,
            keterangan,
        ),
    )

    db.commit()
    id_absensi = cursor.lastrowid

    return {
        "id_absensi": id_absensi,
        "id_mahasiswa": id_mahasiswa,
        "tanggal_absensi": tanggal_absensi,
        "jam_absensi": jam_absensi,
        "keterangan": keterangan,
    }
