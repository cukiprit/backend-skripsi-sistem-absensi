import sqlite3

DATABASE = "database.db"


def create_table():
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()

    cursor.execute(
        """
      CREATE TABLE IF NOT EXISTS mahasiswa(
        NIM TEXT PRIMARY KEY,
        Nama TEXT NOT NULL,
        Divisi TEXT NOT NULL,
        password TEXT NOT NULL
      )
    """
    )

    cursor.execute(
        """
      CREATE TABLE IF NOT EXISTS absensi(
        id_absensi INTEGER PRIMARY KEY AUTOINCREMENT,
        id_mahasiswa TEXT,
        tanggal_absensi TEXT NOT NULL,
        jam_absensi TEXT NOT NULL,
        Keterangan TEXT CHECK(Keterangan IN ("hadir", "tidak hadir")),
        FOREIGN KEY (id_mahasiswa) REFERENCES mahasiswa(NIM)
      )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS notulensi(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          judul TEXT NOT NULL,
          isi_notulensi TEXT NOT NULL,
          tanggal_notulensi TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def get_db():
    connection = sqlite3.connect(DATABASE, check_same_thread=False)
    try:
        yield connection
    finally:
        connection.close()
