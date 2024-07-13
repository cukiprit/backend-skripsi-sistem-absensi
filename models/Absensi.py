from datetime import date, time
from pydantic import BaseModel


class Absensi(BaseModel):
    id_absensi: int = None
    id_mahasiswa: str
    tanggal_absensi: str
    jam_absensi: str
    keterangan: str
