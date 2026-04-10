# Week 1: Mengubah Cara Berpikir (Logika vs. Prediksi)

## 🧠 Inti Konsep
AI menyelesaikan ketidakpastian (*ambiguity*) menggunakan **Probabilitas**.

- **Logika Aturan (Rule-Based)** menggunakan struktur `if-else`. Sangat kaku. Jika ada skenario yang belum ditulis kodenya, sistem akan error.

`Tips: Bayangkan seperti resep masakan tradisional yang kaku. Kita memberi tahu komputer "Jika A, maka lakukan B" (struktur if-else). Sistem ini tidak bisa menangani situasi di luar aturan yang tertulis.`

- **Logika Prediksi (AI):** Berakar pada **Aljabar Linear**. Mesin melihat pola dari jutaan data dan memberikan jawaban yang "paling mungkin benar".

`Tips: AI bekerja menggunakan "Logika Kreasi" yang berakar pada aljabar linear, kalkulus, dan statistik. Alih-alih mengikuti aturan manusia, AI melihat jutaan contoh data untuk mencari hasil yang paling mungkin benar.`



## 🔗 Hubungan Antar Ilmu
- **Ekonomi.** AI membantu pengambilan keputusan investasi di tengah pasar yang tidak menentu.
- **Matematika.** Operasi matriks (perkalian baris dan kolom) adalah "otak" di balik setiap prediksi.

## 📝 PyTorch Cheat-sheet (Basics)
| Fungsi | Kegunaan |
| :--- | :--- |
| `torch.tensor([1, 2])` | Membuat array dasar (tensor) di PyTorch. |
| `tensor.shape` | Melihat dimensi data (ukuran baris/kolom). |
| `torch.matmul(A, B)` | Perkalian matriks (dasar dari semua model AI). |

## 💻 Logika Prediksi Sederhana
```python
import torch

# Representasi fitur (misal: jam belajar & jumlah kopi)
X = torch.tensor([5.0, 2.0]) 

# Bobot (Weight) yang dipelajari AI secara otomatis nanti
W = torch.tensor([0.5, 0.1]) 
bias = 0.5

# Prediksi: (X * W) + bias
y_pred = torch.dot(X, W) + bias
print(f"Hasil Prediksi Kelulusan: {y_pred.item()}")
```

## 🎮 Saatnya bermain!
> “Fun does not come in sizes” – Bart Simpson
![Bart Simpson](../Assets/week1_bartsimpson.png)

> “Kesenangan tidak mengenal ukuran” – Bart Simpson

https://quickdraw.withgoogle.com/