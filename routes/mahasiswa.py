import sqlite3

from fastapi import APIRouter, HTTPException, Depends
from utils.db_setup import get_db
from utils.auth import get_current_user
from models.Mahasiswa import Mahasiswa

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/")
async def read_mahasiswa(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT NIM, Nama, Divisi FROM mahasiswa")
    rows = cursor.fetchall()

    return [{"NIM": row[0], "Nama": row[1], "Divisi": row[2]} for row in rows]


@router.get("/{nim}")
async def read_mahasiswa_by_nim(nim: str, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT NIM, Nama, Divisi FROM mahasiswa WHERE NIM = ?", (nim,))
    row = cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Mahasiswa not found")

    return {"NIM": row[0], "Nama": row[1], "Divisi": row[2]}


@router.post("/", response_model=Mahasiswa)
async def create_mahasiswa(
    mahasiswa: Mahasiswa, db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()

    try:
        cursor.execute(
            "INSERT INTO mahasiswa (NIM, Nama, Divisi) VALUES (?, ?, ?)",
            (mahasiswa.NIM, mahasiswa.Nama, mahasiswa.Divisi),
        )

        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=404, detail="NIM already exists")

    return mahasiswa


@router.put("/{nim}", response_model=Mahasiswa)
async def update_mahasiswa(
    mahasiswa: Mahasiswa, nim: str, db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    cursor.execute(
        "UPDATE mahasiswa SET Nama = ?, Divisi = ? WHERE NIM = ?",
        (mahasiswa.Nama, mahasiswa.Divisi, nim),
    )

    db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Mahasiswa not found")

    return mahasiswa


@router.delete("/{nim}")
async def delete_mahasiswa(nim: str, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("DELETE FROM mahasiswa WHERE NIM = ?", (nim,))
    db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Mahasiswa not found")

    return {"detail": "Mahasiswa deleted successfully"}
