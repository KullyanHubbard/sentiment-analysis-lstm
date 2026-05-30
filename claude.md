# Prediksi Kurs Rupiah - LSTM

## Data
- Source: yfinance
- Ticker: USDIDR=X
- Start: 2001-06-28
- End: dinamis — (datetime.today() + timedelta(days=1)), selalu inklusif hari ini
- Total rows: ~6288+ (bertambah seiring waktu)
- Kolom yang dipakai: Close (+ High, Low untuk hl_range)
- Note: flatten multi-level columns dengan df.columns = df.columns.droplevel(1)
- Cleaning: tick rusak yfinance (nilai 0 / spike <60% atau >200% median lokal 15-hari) → NaN → ffill/bfill

## Fitur Input (20 total)
### Teknikal (9)
- close_diff, return, hl_range, rsi_14, ema_7_dev, ema_21_dev, rolling_std_7, dow, month

### Fundamental (11) — semua di-shift hindari lookahead
- dxy_ret_t1 (DXY via yfinance DX-Y.NYB; return t-1)
- vix_ret_t1 (VIX via yfinance ^VIX; return t-1)
- brent_ret_t1 (Brent crude via yfinance BZ=F)
- ihsg_ret_t1 (IHSG via yfinance ^JKSE)
- tnx_ret_t1 (US 10Y Treasury Yield via yfinance ^TNX)
- cpo_ret_t1 (CPO/Palm Oil via yfinance KCE=F)
- bi_rate, fed_rate, us_cpi, id_cpi, rate_spread (via FRED public CSV, tanpa API key; shift 30 hari)

## Stack
- TensorFlow/Keras
- Python 3.12
- sklearn (MaxAbsScaler)
- CPU (Ubuntu)

## Arsitektur Model
```
Input(WINDOW=30, N_FEATURES=20)
→ LSTM(64, dropout=0.2, recurrent_dropout=0.3, l2=1e-4, return_sequences=True)
→ Dropout(0.3)
→ LSTM(32, dropout=0.2, recurrent_dropout=0.3, l2=1e-4)
→ Dropout(0.3)
→ Dense(HORIZON=7)
```
- Loss: MSE + 0.3 × directional penalty `relu(-y_true * y_pred)`
- Strategi: direct multi-output (7 hari sekaligus, bukan autoregressive)
- Optimizer: Adam | EarlyStopping(patience=10) | ReduceLROnPlateau(patience=5)
- Epochs: max 100

## Scaling — Aturan Penting (teori ML)
- Scaler: **MaxAbsScaler** (bukan MinMaxScaler)
  - Alasan: nol & tanda terjaga → `relu(-y_true*y_pred)` directional penalty sah secara matematis
  - Inverse: `arr * scale_[0]` (tanpa offset, karena MaxAbs tidak menggeser nol)
- **No data leakage**: scaler di-`fit` HANYA pada baris training (`feat[:split+WINDOW]`), lalu `transform` semua
- Walk-Forward rigor: scaler di-**refit per fold** pada `feat[:train_end+WINDOW]` — tidak ada future data masuk scaler

## Catatan Kinerja (penting untuk presentasi)
- Theil's U ≈ 1.0 → model ≈ naive random walk; MAE/MAPE kecil BUKAN bukti keunggulan, melainkan konsekuensi sifat kurs.
- Directional accuracy ~50–55% (walk-forward bisa <50%), marginal di atas acak & bervariasi antar-run.
- Nilai guna: kuantifikasi error & rentang ketidakpastian (MC Dropout), bukan sinyal trading. Summary.txt memuat catatan kejujuran ini.

## Pipeline Cell-by-Cell
| Cell | Isi |
|------|-----|
| 1 | Setup, import, download USDIDR=X (+ bersihkan tick rusak) + DXY/VIX/Brent/IHSG/TNX/CPO/FRED, merge fitur fundamental |
| 2 | Plot historis kurs USD/IDR |
| 3 | Feature engineering 20 fitur; hitung `split`; fit **MaxAbsScaler** pada train-only; buat `features_scaled`, `target_scaled`, `inverse_close_diff()` |
| 4 | Buat sequences (`X`/`y` scaled, `X_raw`/`y_raw` mentah); Walk-Forward Validation dengan **scaler refit per fold**; tentukan `X_train/X_test/y_train/y_test` |
| 5 | Bangun & latih model final (pakai global `features_scaled`) |
| 6 | Evaluasi test set — MAE/RMSE per hari horizon |
| 6b | Metrik profesional: MAPE, Directional Accuracy, Theil's U per horizon |
| 7 | Backtest 7 hari terakhir (direct) + proyeksi 7 hari ke depan (MC Dropout 500x, CI 5%-95%) |
| 8 | Simpan summary.txt (+ metrik pro & walk-forward & catatan kejujuran) + tabel_proyeksi_*.csv; daftar 9 file output |

## Output per Run (folder Result/YYYYMMDD_HHMMSS/)
1. `_historis_kurs_usdidr.png`
2. `_walkforward.png`
3. `_loss_training.png`
4. `_prediksi_vs_aktual_testset.png`
5. `_metrics_professional.png`
6. `_backtest_7hari.png`
7. `_proyeksi_7hari_ke_depan.png`
8. `_tabel_proyeksi_7hari.csv`
9. `_summary.txt`

## Konstanta Global
| Variabel | Nilai | Keterangan |
|----------|-------|-----------|
| WINDOW | 30 | hari lookback |
| HORIZON / FORECAST_DAYS | 7 | hari prediksi ke depan |
| SKIP | 30 | warmup RSI/EMA, dikecualikan dari feat |
| MC_SAMPLES | 500 | sampel MC Dropout untuk CI |

## Instruksi Otomatis
- JANGAN PERNAH menjalankan notebook secara otomatis. User yang akan menjalankan Run All sendiri di IDE.
- Jangan panggil `jupyter nbconvert --execute`, `python -m jupyter`, atau perintah eksekusi notebook lainnya tanpa diminta eksplisit.
- Setelah selesai update cell, beritahu user bahwa kode siap di-Run All — jangan dieksekusi.
- Install dependencies yang kurang otomatis dengan pip (ini boleh, hanya eksekusi notebook yang dilarang).
- Untuk perubahan terkait ML (scaling, loss, split, validasi): jelaskan teori yang benar dulu, lalu **tanya user** sebelum menerapkan.
