import sqlite3

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from utils.db_setup import get_db


security = HTTPBasic()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def authenticate_user(db: sqlite3.Connection, NIM: str, password: str):
    cursor = db.cursor()
    cursor.execute("SELECT password, Divisi FROM mahasiswa WHERE NIM = ?", (NIM,))
    row = cursor.fetchone()
    if row is None:
        return False
    hashed_password = row[0]
    role = row[1]
    if not verify_password(password, hashed_password):
        return None
    return {"NIM": NIM, "role": role}


async def get_current_user(
    credentials: HTTPBasicCredentials = Depends(security),
    db: sqlite3.Connection = Depends(get_db),
):
    if not authenticate_user(db, credentials.username, credentials.password):
        raise HTTPException(status_code=401, detail="Invalid Credentials")
    return credentials.username


# async def get_current_user(
#     credentials: HTTPBasicCredentials = Depends(security),
#     db: sqlite3.Connection = Depends(get_db),
# ):
#     if not authenticate_user(db, credentials.username, credentials.password):
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid username or password",
#             headers={"WWW-Authenticate": "Basic"},
#         )
#     return credentials.username
