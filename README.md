# MLOps--Dota-Draft-

Repositori ini berisi pipeline awal MLOps untuk analisis draft hero Dota menggunakan data dari OpenDota API.
Implementasi utama saat ini ada di Zone A (Ingestion dan Patch Watcher) untuk menghasilkan dataset training berbobot patch.

## Tujuan Proyek

- Mengambil data match `pro` dan `public` dari OpenDota.
- Melakukan parsing hero draft 5v5 dan metadata match.
- Memberi bobot instance berdasarkan kedekatan patch.
- Menyimpan dataset hasil ingestion ke format CSV untuk tahap training.

## Struktur Direktori

Struktur mengikuti konvensi Cookiecutter Data Science:

```text
.
|-- .devcontainer/            # Konfigurasi GitHub Codespaces
|-- config/                   # Konfigurasi proyek
|-- data/
|   |-- external/             # Data pihak ketiga
|   |-- interim/              # Data hasil transformasi sementara
|   |-- processed/            # Data final untuk modeling
|   `-- raw/                  # Data mentah yang immutable
|-- docs/                     # Dokumentasi tambahan
|-- models/                   # Artefak model terlatih
|-- notebooks/                # Notebook eksperimen
|-- references/               # Data dictionary, catatan, referensi
|-- reports/
|   `-- figures/              # Visualisasi untuk laporan
|-- src/
|   `-- dota_draft/
|       |-- config.py         # Helper konfigurasi
|       |-- dataset.py        # Zone A: ingestion OpenDota + patch weighting
|       |-- features.py       # Feature engineering
|       |-- plots.py          # Utility visualisasi
|       `-- modeling/
|           |-- train.py      # Training model
|           `-- predict.py    # Inference model
|-- .gitignore                # Python gitignore
|-- LICENSE                   # MIT License
`-- requirements.txt          # Dependensi Python
```

## Implementasi Zone A

Modul `src/dota_draft/dataset.py` mengimplementasikan:

- `fetch_patch_mapping()`: mapping patch ID OpenDota ke versi semantik.
- `fetch_public_matches()` dan `fetch_pro_matches()`: koleksi listing match.
- `fetch_match_details()`: ambil detail match dengan retry/backoff saat rate limit.
- `parse_match_data()`: ekstraksi draft hero Radiant vs Dire (5v5).
- `get_weight()`: bobot data berdasarkan jarak patch.
- `prepare_training_dataframe()`: membentuk DataFrame siap training.
- `run_ingestion()`: orkestrasi end-to-end dan simpan output CSV.

Output default ingestion:

- `data/processed/training_data_weighted.csv`

## Menjalankan di GitHub Codespaces

1. Buka repositori di GitHub.
2. Klik `Code` -> `Codespaces` -> `Create codespace on main`.
3. Tunggu proses build environment selesai.
4. Verifikasi Python:

```bash
python --version
```

5. Install dependensi:

```bash
pip install -r requirements.txt
```

6. Jalankan ingestion pipeline:

```bash
python -m src.dota_draft.dataset
```

Konfigurasi default Codespaces pada repo ini:

- Python `3.11`
- Ekstensi: Python, Pylance, Jupyter, Ruff

## Ringkasan Data Output

Kolom yang dihasilkan mencakup:

- Metadata: `match_id`, `match_date`, `start_time`, `patch`, `duration`, `match_type`
- Target: `radiant_win`
- Draft picks: `radiant_hero_1..5`, `dire_hero_1..5`
- Context: `series_id` (pro) atau `avg_mmr` (public)
- Weighting: `weight`