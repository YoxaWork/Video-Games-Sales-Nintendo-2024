# Analisis dan Prediksi Penjualan Game Nintendo

Tugas Machine Learning: data splitting, data cleaning, EDA, dan modeling pada data penjualan video game konsol Nintendo (Wii, DS, 3DS, Wii U, Switch). Target yang diprediksi adalah `total_sales` (juta kopi), sehingga kasus ini adalah **regresi**.

## Dataset

- **Nama:** Video Game Sales Nintendo 2024
- **Sumber:** Kaggle (asaniczka), data asli dari VGChartz
- **Link:** https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024
- **Data yang dipakai:** subset konsol Nintendo, X baris (isi dengan jumlah asli)
- **Pembagian data:** 80% train dan 20% test (`random_state=1`)

## Alur Pengerjaan

1. **Data splitting:** ambil konsol Nintendo, buang duplikat dan baris tanpa target, lalu bagi menjadi train dan test dan simpan sebagai `train.csv` dan `test.csv`
2. **Data loading:** muat kembali `train.csv` dan `test.csv`
3. **Data cleaning:** cek informasi dataset dan missing value, hapus kolom dengan missing value di atas 75%, isi nilai kosong (median untuk numerik, "Unknown" untuk kategori), cek duplikat, tangani outlier dengan metode IQR
4. **EDA dan ExDA:** histogram, distribusi, penjualan per konsol dan genre, tren per tahun, matriks korelasi, dan kesimpulan
5. **Transformasi:** One-Hot Encoding dan standardisasi (`StandardScaler`), dihitung dari data train saja
6. **Modeling dan evaluasi:** Lars, Linear Regression, dan Gradient Boosting Regressor, dengan baseline (median) sebagai pembanding
7. **Metrik evaluasi:** MAE, MSE, dan R2 (RMSE sebagai tambahan)

## Catatan Metode

- Duplikat dibuang **sebelum** split agar baris kembar tidak masuk ke train dan test sekaligus (mencegah data leakage).
- Semua nilai yang dipelajari dari data (median, batas outlier, daftar publisher populer, parameter standardisasi) dihitung dari data train, lalu diterapkan ke data test.
- Outlier pada fitur ditangani dengan IQR. Outlier pada target (`total_sales`) dipertahankan karena merupakan game best-seller yang valid.
- Kolom `na_sales`, `jp_sales`, `pal_sales`, dan `other_sales` tidak dipakai sebagai fitur karena jumlahnya sama dengan target.

## Hasil

Evaluasi pada data test (20% data):

| Model | MAE | MSE | R2 |
|---|---|---|---|
| Baseline (median) | 0.2017 | 0.2017 | -0.1048 |
| Lars | 0.2337 | 0.1810 | 0.0089 |
| Linear Regression | 0.2130 | 0.1656 | 0.0932 |
| Gradient Boosting Regressor | 0.2132 | 0.1642 | 0.1008 |

**Model terbaik:** Gradient Boosting Regressor, karena memiliki R2 tertinggi (0.1008) dan MSE terendah (0.1642).

**Interpretasi:**
- Gradient Boosting dan Linear Regression lebih baik dari baseline pada MSE dan R2. Baseline memiliki R2 negatif, artinya prediksinya lebih buruk daripada sekadar menebak rata-rata.
- Baseline memiliki MAE terkecil karena median memang nilai yang meminimalkan kesalahan absolut. Model lain dilatih meminimalkan kesalahan kuadrat, sehingga lebih baik pada MSE dan R2.
- Nilai R2 masih rendah (sekitar 0.10), artinya fitur yang tersedia (konsol, genre, publisher, tahun rilis) hanya menjelaskan sebagian kecil variasi penjualan. Penjualan game dipengaruhi hal yang tidak ada di data, seperti marketing dan popularitas franchise.
- Lars memiliki hasil paling lemah karena dibatasi hanya satu fitur (`n_nonzero_coefs=1`).

## File

| File | Isi |
|---|---|
| `Video_Game_Sales_2024.ipynb` | Notebook utama |

File berikut dibuat otomatis saat notebook dijalankan: `train.csv`, `test.csv`, `train_bersih.csv`, `test_bersih.csv`, dan `model_terbaik.joblib`.

## Cara Menjalankan

1. Download `vgchartz-2024.csv` dari link Kaggle di atas.
2. Buka `Video_Game_Sales_2024.ipynb` di Google Colab.
3. Upload `vgchartz-2024.csv` ke folder Colab.
4. Jalankan semua cell dari atas ke bawah (Runtime, lalu Run all).
