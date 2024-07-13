from pydantic import BaseModel


class AbsensiMahasiswa(BaseModel):
    id_absensi: int
    id_mahasiswa: str
    tanggal_absensi: str
    jam_absensi: str
    keterangan: str
    nim: str
    nama: str
    divisi: str
