from pydantic import BaseModel


class Notulensi(BaseModel):
    judul: str
    isi_notulensi: str
    tanggal_notulensi: str
