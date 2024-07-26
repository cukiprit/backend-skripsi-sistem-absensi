import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from utils.db_setup import get_db
from utils.auth import get_password_hash, authenticate_user

from models.User import User

router = APIRouter()
security = HTTPBasic()


@router.post("/register")
async def register_user(user: User, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()

    try:
        hashed_password = get_password_hash(user.password)
        cursor.execute(
            "INSERT INTO mahasiswa(NIM, Nama, Divisi, password) VALUES (?, ?, ?, ?)",
            (user.NIM, user.Nama, user.Divisi, hashed_password),
        )
        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="NIM Already exists")
    return {"detail": "User registered successfully"}


@router.post("/login")
async def login(
    credentials: HTTPBasicCredentials = Depends(security),
    db: sqlite3.Connection = Depends(get_db),
):
    if not authenticate_user(db, credentials.username, credentials.password):
        raise HTTPException(status_code=401, detail="Invalid Credentials")
    return {"detail": "Login successful"}
