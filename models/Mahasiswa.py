from pydantic import BaseModel


class Mahasiswa(BaseModel):
    NIM: str
    Nama: str
    Divisi: str
