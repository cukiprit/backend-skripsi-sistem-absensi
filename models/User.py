from pydantic import BaseModel


class User(BaseModel):
    NIM: str
    Nama: str
    Divisi: str
    password: str
