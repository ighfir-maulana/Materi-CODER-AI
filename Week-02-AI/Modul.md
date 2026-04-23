# 🤖 Modul AI: Logika vs Prediksi

> [!NOTE]
> **Tujuan Pembelajaran:** Memahami perbedaan antara antara beberapa tipe data yang ada, seperti regression, klasifikasi dan clustering. Setelah itu kiat akan masuk lebih mendalam soal pembahasan ANN, apa bedanya dengan perceptron, serta bagaimana mekansime / cara kerjanya dalam logika matematika dan juga dalam program nya. 

---

## 1. 🧠 Tipe Data Dalam AI

AI mempunyai sebuah tipe data nya sendiri, tipe data ini adalah sebuah masalah yang berbeda beda di dunia nyata, setidaknya ada 2 tipe data yang paling umum:

### 1. Supervised Learning

Supervised Learning adalah pembelajaran yang perlu data berlabel, lalu mesin bakal belajar dari label itu untuk mendapatkan polanya. 

### 2. Unsupervised Learning

Unsupervised Learning adalah pembelajaran yang tidak perlu data berlabel, lalu mesin bakal belajar dari data tersebut untuk mendapatkan polanya. 

![Cara Kerja AI](./Assets/9.jpg)

Supervised learning = belajar pakai mentor (ada petunjuk labelnya / ada refrensi mana yang benar dan salah)

Unsupervised learning = Belajar Mandiri (ga ada petunjuk labelnya)


Dari Kedua jenis tipe pembelajarannya, kita bisa memecah problem dalam dunia nyata menjadi 3 hal, yaitu:

1. Regresi
2. Klasifikasi
3. Clustering 

mari kita bahas dulu tipe data yang paling umum yaitu regresi dan klasifikasi. 

### 1. Regresi (Regression)

Regresi adalah salah satu jenis masalah dalam sebuah machine learning dan juga AI. regresi biasnya di gunakan dalam masalah yang outputnya berupa angka / bilangan kontinu. contohnya:

- **Prediksi Harga Rumah**
![Cara Kerja AI](./Assets/1.jpeg)

- **Prediksi Nilai Ujian berdasarkan lamanya belajar**
![Cara Kerja AI](./Assets/2.png)

- **Prediksi Harga bitcoin**
![Cara Kerja AI](./Assets/3.jpg)

---

Regresi termasuk kedalam supervising learning (Pembelajaran yang terawasi), artinya kita perlu memberikan label terhadap data kita agar mesin mampu mengenali pola dari data tersebut. 

###2. Klasifikasi (Classification)

Klasifikasi adalah jenis masalah yang mengharuskan kita untuk mengkategorikan sesuatu atau menebak sesuatu berdasarkan kategori yang sudah ditentukan. Contohnya:

- **Prediksi apakah email ini spam atau bukan**
![Cara Kerja AI](./Assets/4.png)

- **Prediksi apakah gambar ini kucing atau anjing**
![Cara Kerja AI](./Assets/5.jpg)

Robot termasuk salah satu yang sangat bergantung dengan algoritma klasifikasi, agar mampu mengenali lingkungan sekitarnya.

![Cara Kerja AI](./Assets/6.jpeg)

---

Klasifikasi termasuk kedalam supervising learning (Pembelajaran yang terawasi), artinya kita perlu memberikan label terhadap data kita agar mesin mampu mengenali pola dari data tersebut. 

### 3. Clustering (Pengelompokan)

Clustering adalah jenis masalah yang mengharuskan kita untuk mengelompokkan sesuatu berdasarkan kemiripan karakteristiknya. Contohnya:

- **Segmentasi pelanggan**
![Cara Kerja AI](./Assets/7.jpg)

- **Pengelompokan berita berdasarkan topik**
![Cara Kerja AI](./Assets/8.jpeg)

---

Clustering termasuk kedalam unsupervised learning (Pembelajaran yang tidak terawasi), artinya kita tidak perlu memberikan label terhadap data kita. AI akan mengelompokkan data berdasarkan kemiripan karakteristiknya. 


> **Jadi, kalau kita mau membuat AI, Tentukan dulu jenis masalahnya, apakah regresi, klasifikasi, atau justru clustering. baru deh dari situ kta tentukan algoritma apa yang cocok untuk masalah tersebut.**


Tipe data dan Jenis masalah yang berbeda menentukan bagaimana Input dan output dari AI tersebut. 

### Untuk INPUT

#### 1. Data Kontinu (Harga Rumah): 
Inputnya cuma beberapa kolom (Luas, Lokasi). Jadi Input Layer-nya kecil (misal 2 neuron).

#### 2. Data Tak Terstruktur (Gambar): 
Inputnya adalah pixel. Kalau gambar dari ESP32-CAM kamu ukurannya 128 x 128, maka Input Layer-nya harus punya 16.384 neuron!


### Untuk OUTPUT

#### 1. Jika tugasnya Klasifikasi (Kucing vs Anjing): 
Outputnya harus kategori. Kita butuh fungsi aktivasi seperti Softmax atau Sigmoid agar hasilnya jadi probabilitas (0 sampai 1).

// tambahkan gamabr ilustrasi juag boleh

#### 2. Jika tugasnya Regresi (Harga Rumah): 
Outputnya angka bebas (bisa ratusan juta). Kita tidak boleh pakai Sigmoid di akhir, karena Sigmoid akan membatasi angka maksimal jadi 1. Kita butuh fungsi Linear.


---

## 2. ANN

Artificial Neural Network yang merupakan cabang dari machine learning. ANN ini terinspirasi dari cara kerja sistem saraf manusia. 


Link Simulasi: https://enki1030.github.io/Simulasi_ANN/


---

### 3. Bagaimana Cara kerja ANN 

Cara kerja ANN sebenarnya cukup mudah untuk di pahami.

ANN terdiri dari 3 lapisan utama:

#### Input Layer -> Hidden Layer -> Output Layer

3 Layer ini akan melalui 2 proses utama:

- **1. Feed Forward / Forward Pass**
- **2. Backpropagation**


#### 1. Feed Forward / Forward Pass

![Cara Kerja AI](./Assets/10.gif)

Feed Forward adalah proses perhitungan matematika yang dlakukan ANN untuk memprediksi atau mendapatkan hasil (Output) dari inputan. 

#### 1. Input Layer

Tugas nya untuk menerima inputan, inputanya bisa berupa angka atau gambar. 

- Data Angka (Prediski Harga HP)
![Cara Kerja AI](./Assets/11.png)


- Data Gambar (Prediksi Anjing)
![Cara Kerja AI](./Assets/12.png)

#### 2. Hidden Layer

Tugas tergantung, kalau di traning, tugas nya adalah untuk melakukan esktraksi data dan pola agar dengan cara menyesuaikan bobotnya dan biasnya (bagian Backpropagation).

kalau dalam tes, tugasnya adalah untuk mencocokkan inputan tersebut apakah sesuai dengan pola yang sudah di traning atau tidak. (Bagian Forward Pass)

> **(Pembahasan lebih lanjut feed forward melalui sebuah tes nantinya. Di sini di jelasin mekanisme mulai dari rumus yang di gunakan sampai Fungsi aktivasi yang biasanya di gunakan.)
Di sini akan berisi rumus, tempat dilaukannay perhitungan matematika, Logika pemrogramannya. 

#### 3. Output Layer

Output Layer adalah Lapisan terakhir yang mengembalikan hasil akhir dari pemrosesan Hidden Layer.


#### 2. Backpropagation

![Cara Kerja AI](./Assets/13.gif)

Backpropagation adalah proses penyesuaian bobot dan bias pada ANN agar menghasilkan output yang lebih akurat.

ada dua langkah dalam backpropagation, yaitu:

1. **Hitung Error:**

![Cara Kerja AI](./Assets/14.png)

Error dihitung dengan membandingkan output yang dihasilkan ANN dengan output yang seharusnya (label).

2. **Hitung gradient**

![Cara Kerja AI](./Assets/15.png)

Gradient adalah perubahan nilai Error berdasarkan perubahan bobot dan bias. 

3. **Update Bobot dan Bias:**

![Cara Kerja AI](./Assets/16.png)

Bobot dan bias diupdate agar hasil prediksinya semakin bagus. Proses ini di ulangi berkali-kali sampai Error mencapai ambang batas tertentu.


---

### 4. Perbedaan ANN dan Perceptron

ANN dan Perceptron adalah dua hal yang berkaitan namun memiliki perbedaan yang cukup signifikan.

Perbedaan ANN dan Perceptron terletak pada jumlah Hidden Layer yang dimilikinya. ANN memiliki lebih dari satu Hidden Layer, sedangkan Perceptron hanya memiliki satu Hidden Layer.

![Cara Kerja AI](./Assets/17.png)

---

### 5. Implementasi ANN dalam Program

> Praktikum
