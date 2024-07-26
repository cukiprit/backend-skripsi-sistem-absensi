from fastapi.middleware.cors import CORSMiddleware
from utils.db_setup import create_table

from fastapi import FastAPI
from routes.absensi import router as AbsensiRouter
from routes.mahasiswa import router as MahasiswaRouter
from routes.auth import router as AuthRouter
from routes.notulensi import router as NotulensiRouter

origins = ["http://localhost", "http://localhost:5173"]

app = FastAPI()
create_table()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(MahasiswaRouter, prefix="/api/mahasiswa", tags=["mahasiswa"])
app.include_router(AbsensiRouter, prefix="/api/absensi", tags=["absensi"])
app.include_router(AuthRouter, prefix="/api", tags=["auth"])
app.include_router(NotulensiRouter, prefix="/api/notulensi", tags=["notulensi"])
