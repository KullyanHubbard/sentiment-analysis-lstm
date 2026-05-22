# Prediksi Kurs Rupiah - LSTM

## Data
- Source: yfinance
- Ticker: USDIDR=X
- Start: 2001-06-28
- End: dinamis — (datetime.today() + timedelta(days=1)), selalu inklusif hari ini
- Total rows: ~6284+ (bertambah seiring waktu)
- Kolom yang dipakai: Close
- Note: flatten multi-level columns dengan df.columns = df.columns.droplevel(1)

## Stack
- TensorFlow/Keras
- Python 3.12
- CPU (Ubuntu)