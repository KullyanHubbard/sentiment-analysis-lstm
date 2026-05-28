# Prediksi Kurs Rupiah - LSTM

## Data
- Source: yfinance
- Ticker: USDIDR=X
- Start: 2001-06-28
- End: dinamis — (datetime.today() + timedelta(days=1)), selalu inklusif hari ini
- Total rows: ~6288+ (bertambah seiring waktu)
- Kolom yang dipakai: Close (+ High, Low untuk hl_range)
- Note: flatten multi-level columns dengan df.columns = df.columns.droplevel(1)

## Fitur Input (20 total)
### Teknikal (9)
- close_diff, return, hl_range, rsi_14, ema_7_dev, ema_21_dev, rolling_std_7, dow, month

### Fundamental (11) — semua di-shift(1) hindari lookahead
- dxy_return, dxy_lag1 (DXY via yfinance DX-Y.NYB)
- vix_return, vix_lag1 (VIX via yfinance ^VIX)
- brent_return (Brent crude via yfinance BZ=F)
- ihsg_return (IHSG via yfinance ^JKSE)
- bi_rate, fed_rate, us_cpi, id_cpi, rate_spread (via FRED public CSV, tanpa API key)

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
→ MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.1)
→ LayerNormalization (dengan residual Add)
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

## Pipeline Cell-by-Cell
| Cell | Isi |
|------|-----|
| 1 | Setup, import, download USDIDR=X + DXY/VIX/Brent/IHSG/FRED, merge fitur fundamental |
| 2 | Plot historis kurs USD/IDR |
| 3 | Feature engineering 20 fitur; hitung `split`; fit **MaxAbsScaler** pada train-only; buat `features_scaled`, `target_scaled`, `inverse_close_diff()` |
| 4 | Buat sequences (`X`/`y` scaled, `X_raw`/`y_raw` mentah); Walk-Forward Validation dengan **scaler refit per fold**; tentukan `X_train/X_test/y_train/y_test` |
| 5 | Bangun & latih model final (pakai global `features_scaled`) |
| 6 | Evaluasi test set — MAE/RMSE per hari horizon |
| 6b | Metrik profesional: MAPE, Directional Accuracy, Theil's U per horizon |
| 7 | Backtest 7 hari terakhir (direct) + proyeksi 7 hari ke depan (MC Dropout 500x, CI 5%-95%) |
| 7b | Interpretasi CI berwarna per confidence level (Tinggi/Sedang/Rendah) |
| 8 | Simpan summary.txt + tabel_proyeksi_*.csv; daftar 10 file output |

## Output per Run (folder Result/YYYYMMDD_HHMMSS/)
1. `_historis_kurs_usdidr.png`
2. `_walkforward.png`
3. `_loss_training.png`
4. `_prediksi_vs_aktual_testset.png`
5. `_metrics_professional.png`
6. `_backtest_7hari.png`
7. `_proyeksi_7hari_ke_depan.png`
8. `_proyeksi_ci_enhanced.png`
9. `_tabel_proyeksi_7hari.csv`
10. `_summary.txt`

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
