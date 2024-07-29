import sqlite3

from fastapi import APIRouter, HTTPException, Depends
from utils.db_setup import get_db
from utils.auth import get_current_user
from models.Notulensi import Notulensi

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/")
async def read_notulensi(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute(
        "SELECT id, judul, isi_notulensi, DATE(tanggal_notulensi) FROM notulensi"
    )

    rows = cursor.fetchall()

    return [
        {"id": row[0], "judul": row[1], "isi": row[2], "tanggal": row[3]}
        for row in rows
    ]


@router.get("/{id}")
async def read_notulensi_by_id(id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute(
        "SELECT id, judul, isi_notulensi, DATE(tanggal_notulensi) FROM notulensi WHERE id = ?",
        (id,),
    )

    row = cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Notulensi not found")

    return [{"id": row[0], "judul": row[1], "isi": row[2], "tanggal": row[3]}]


@router.post("/", response_model=Notulensi)
async def create_notulensi(
    notulensi: Notulensi, db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()

    cursor.execute(
        "INSERT INTO notulensi (judul, isi_notulensi, tanggal_notulensi) VALUES (?, ?, ?)",
        (notulensi.judul, notulensi.isi_notulensi, notulensi.tanggal_notulensi),
    )

    db.commit()

    return notulensi


@router.put("/{id}", response_model=Notulensi)
async def update_notulensi(
    notulensi: Notulensi, id: int, db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    cursor.execute(
        "UPDATE notulensi SET judul = ?, isi_notulensi = ?, tanggal_notulensi = ? WHERE id = ?",
        (notulensi.judul, notulensi.isi_notulensi, notulensi.tanggal_notulensi, id),
    )

    db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Notulensi not found")

    return notulensi


@router.delete("/{id}")
async def delete_notulensi(id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("DELETE FROM notulensi WHERE id = ?", (id,))

    db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Notulensi not found")

    return {"detail": "Notulensi deleted successfully"}
