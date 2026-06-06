🧠 MINGGU 6: ATTENTION MECHANISM & ARSITEKTUR TRANSFORMER

Topik: "Attention Is All You Need" dan Fondasi Model Bahasa Raksasa (LLM)

---

## 1. ERA SEBELUM TRANSFORMER: MENGAPA KITA BUTUH "PERHATIAN"?

### A. Keterbatasan Ingatan Mesin (The Bottleneck)

Pengolahan bahasa alami atau *natural language processing* (NLP) adalah bidang yang berfokus pada manipulasi dan pemahaman teks oleh komputer. Sebelum transformer, bidang ini didominasi oleh model sekuensial. Mari kita memutar waktu sejenak. Dulu, untuk membuat AI yang bisa memahami teks, para ilmuwan sangat bergantung pada arsitektur bernama **RNN (Recurrent Neural Network)** dan versi mutakhirnya, **LSTM (Long Short-Term Memory)**. 

Cara kerja RNN ini dibuat sangat mirip dengan cara manusia membaca: berurutan dari kiri ke kanan, kata per kata. AI akan membaca kata pertama, memprosesnya, menyimpan informasinya, lalu lanjut membaca kata kedua, dan seterusnya secara berurutan.

![RNN](.\assets\1.jpg)
Kedengarannya sangat masuk akal dan natural, bukan? Namun, pada praktiknya metode ini ternyata memendam kelemahan yang sangat fatal: **The Bottleneck (Leher Botol)**. 

![Bottleneck](./assets/2.png)
Coba bayangkan Anda disuruh membaca buku teks sejarah setebal 1000 halaman secara berurutan. Saat Anda akhirnya sampai di halaman 1000, Anda kemungkinan besar sudah lupa dengan detail nama tokoh atau tempat yang ada di halaman pertama. Inilah persisnya yang dialami oleh model RNN! Semakin panjang kalimat atau dokumen yang diproses, konteks yang berada di awal teks akan semakin memudar dan akhirnya terlupakan oleh mesin.

Secara matematis, fenomena ini disebut sebagai **Masalah Vanishing Gradient**. Di minggu lalu, kita sudah melihat bagaimana arsitektur ResNet menyelesaikan masalah "hilangnya sinyal" ini pada pemrosesan gambar (CNN) menggunakan jalur pintas (*skip connections*). Namun, untuk data teks yang berurutan, informasinya harus mengantre sangat panjang. Sinyal asli dari awal kalimat perlahan "menguap" karena harus melewati ratusan kali perkalian matematis sebelum sampai di akhir. 

Akibatnya? Model menjadi sangat "pelupa" pada teks panjang. Lebih parahnya lagi, pelatihannya berjalan sangat lambat karena harus memproses teks secara sekuensial (harus menunggu kata ke-1 selesai untuk bisa memproses kata ke-2) dan tidak bisa diparalelkan di banyak GPU.

*Lalu, jika mesin yang membaca teks kata per kata selalu berujung menjadi pelupa dan lambat, adakah cara lain agar AI bisa memproses ribuan kata sekaligus tanpa melupakan konteks aslinya?*

### B. Lahirnya Konsep "Attention": Terinspirasi dari Otak Manusia

Jawaban dari kebuntuan di atas ternyata datang dari ilmu sains kognitif (*cognitive science*) dan cara kerja **otak manusia** itu sendiri. 

Di dalam otak kita, terdapat mekanisme pemrosesan informasi yang disebut *Visual/Cognitive Attention*. Otak kita tidak pernah memproses semua informasi yang masuk secara merata (karena itu akan membuat otak *overload*). Sebaliknya, otak kita secara otomatis melakukan *Selective Focus* atau fokus selektif. Inilah yang secara langsung menginspirasi para ilmuwan AI untuk menciptakan mekanisme **"Attention" (Perhatian)** pada jaringan saraf tiruan.

Contoh attention
Contoh 1:
Game Find the difference
![Game1](./assets/3.jpg)

Contoh 2:
Game Find the Object
![Game2](./assets/4.jpg)

![Game3](./assets/5.jpeg)
Bayangin saat kita melihat sebuah foto pasar yang ramai untuk mencari teman kita, apakah kita menganalisis setiap piksel dari ujung kiri ke kanan secara merata? Tentu tidak! Kita langsung memberikan **fokus (attention)** pada wajah-wajah orang dan mengabaikan langit atau jalanan. Sama halnya saat menerjemahkan kalimat kompleks, kita memberikan "bobot" lebih pada kata kunci tertentu.

Dalam dunia AI, *Attention Mechanism* adalah sebuah terobosan matematis yang memungkinkan mesin meniru cara otak manusia beroperasi ini. AI tidak lagi dipaksa mengingat seluruh teks secara membabi buta ke dalam satu ruang memori sempit (seperti RNN), melainkan dibekali kemampuan untuk secara dinamis bertanya: *"Saat saya membaca kata saat ini, kata-kata mana saja di masa lalu yang paling relevan dan penting untuk saya beri perhatian lebih?"*

---

## 2. BEDAH TEORI: "ATTENTION IS ALL YOU NEED" DAN KELAHIRAN LLM

### A. Anatomi Self-Attention: Query, Key, dan Value (Q, K, V)

Sebelum kita masuk lebih dalam, kita harus tahu dulu bagaimana mesin memahami teks. Mesin tidak memahami teks secara langsung. Mereka memahami angka. Transformer menggunakan beberapa langkah untuk mengubah kata menjadi format numerik.

#### Tokenisasi
Proses memecah teks menjadi unit-unit kecil yang disebut token.
- **Word-level**. Word-level membagi berdasarkan kata utuh. Kelemahannya adalah risiko *Out of Vocabulary (OOV)* jika kata baru muncul.
- **Subword-level** (paling umum digunakan). Subword-level membagi kata berdasarkan akar kata atau potongan karakter (misal: "*playing*" menjadi "*play*" dan "*ing*"). Ini lebih efisien dan menangani variasi kata dengan baik.
- **Character-level**. Character-level membagi per karakter. Sangat lambat dan sulit menangkap makna semantik.

#### Word Embedding
Setelah menjadi token, setiap token dihubungkan dengan vektor dimensi tinggi (misal: 12.288 dimensi pada GPT-3).
- Makna geometris yaitu kata-kata dengan makna serupa berada di posisi yang berdekatan dalam ruang vektor.
- Operasi matematika yaitu arah dalam ruang vektor ini menyandikan makna semantik. Contoh klasik: "King - Man + Woman = Queen".
- *Static vs. Contextual Embedding* yaitu model lama memberikan vektor tetap untuk satu kata. Transformer menghasilkan *contextual embedding*, di mana vektor kata "bank" akan berbeda jika berada dalam konteks "sungai" dibandingkan dengan konteks "keuangan".

Lalu bagaimana sebenarnya *attention* bekerja secara matematis sebagai jantung transformer? Mekanisme *attention* memungkinkan model untuk menentukan kata mana dalam sebuah kalimat yang paling penting untuk memahami kata tertentu.

Mari gunakan **analogi pencarian YouTube**.
Dalam arsitektur *Self-Attention*, setiap kata yang masuk akan dipecah menjadi tiga representasi vektor: **Query (Q), Key (K), dan Value (V)**.

*   **Query (Q) - "Apa yang sedang dicari"**. Mirip seperti teks yang Anda ketik di kolom pencarian YouTube (misal: "Tutorial AI"). Ini adalah kata yang saat ini sedang diproses oleh model.
*   **Key (K) - "Label/Kategori di Rak"**. Mirip seperti judul, tag, atau deskripsi pada video-video yang ada di YouTube. Ini adalah identitas dari kata-kata lain di dalam kalimat yang sama.
*   **Value (V) - "Isi Video"**. Ini adalah konten sebenarnya dari video tersebut, atau dalam NLP, makna intrinsik dari kata tersebut.

**Cara kerjanya:**
Sistem akan menghitung seberapa mirip dengan skor kesamaan antara **Query** dengan semua **Key** yang ada menggunakan perkalian matriks (*dot product*). Skor ini melewati fungsi **Softmax** (telah kita pelajari di pertemuan sebelumnya) untuk menghasilkan distribusi probabilitas (bobot atensi) yang berjumlah 1. Jika kemiripannya tinggi (skornya besar), maka model akan mengambil **Value** dari kata/token tersebut, dikalikan dengan bobot atensi dan menggabungkannya sebagai konteks. Dengan ini, kata "Bank" bisa mengenali apakah ia merujuk pada institusi keuangan (jika Key di sekitarnya adalah "uang", "bunga") atau tepi sungai (jika Key di sekitarnya adalah "air", "perahu").

![Query Key Value Analogy](https://miro.medium.com/v2/resize:fit:1100/1*S4CGk8HIn73x-4T0HqP6lw.png)

### B. Arsitektur Sang Revolusioner: Transformer

Pada tahun 2017, peneliti Google merilis *paper* legendaris berjudul **"Attention Is All You Need"** tentang arsitektur jaringan saraf revolusioner. 

Penemuan terbesarnya bukanlah menciptakan *Attention*, melainkan penemuan bahwa **kita bisa membuang RNN sepenuhnya**. Mereka membuktikan bahwa kita tidak perlu membaca teks secara berurutan. Dengan menggunakan **Multi-Head Attention**, Transformer melihat **semua kata dalam satu kalimat secara serentak (paralel)**, menghitung hubungan Q, K, V untuk setiap kata dengan setiap kata lainnya di waktu yang bersamaan.

Karena sifatnya yang paralel, Transformer sangat luar biasa cepat jika dilatih di atas GPU. *Bottleneck* kecepatan akhirnya hancur. Inilah titik mula AI melesat begitu cepat, karena sekarang kita bisa melatih model dengan data sebesar internet!

![Arsitektur Transformer](https://miro.medium.com/v2/resize:fit:864/1*BHzGVskWGS_3jEcYYi6miQ.png)
*(Arsitektur asli Transformer dari paper tahun 2017)*

### C. Dari Transformer Menuju ChatGPT, Gemini, dan Claude

Bagaimana arsitektur dari 2017 ini bisa menjadi ChatGPT yang kita kenal sekarang?

Arsitektur Transformer asli memiliki dua bagian: **Encoder** (untuk memahami teks masukan) dan **Decoder** (untuk menghasilkan teks keluaran). 

*Encoder* berfungsi membaca input secara utuh dan membangun representasi kontekstual yang banyak dari teks tersebut. Contoh modelnya adalah BERT.

Sedangkan *decoder* berfungsi menghasilkan urutan *output* satu per satu secara *auto-regresif* berdasarkan representasi dari *encoder*.

Model seperti **ChatGPT (Generative Pre-trained Transformer)** pada dasarnya "membuang" bagian Encoder dan hanya menggunakan **Decoder**. Model ini dilatih dengan satu tugas yang luar biasa sederhana: **Memprediksi Kata Selanjutnya (Next-Token Prediction)** dari milyaran halaman web di internet.

![NextTokenPrediction](./assets/7.jpeg)

Inovasi lainnnya yaitu:
- **Positional Encoding**. Karena Transformer memproses semua kata sekaligus, ia tidak tahu urutan kata secara alami. Contoh: "Bahlil belajar AI" akan terlihat identik dengan "AI belajar Bahlil" bagi model karena keduanya mengandung token yang sama. Oleh karena itu, informasi posisi harus "disuntikkan" ke dalam data itu sendiri. Informasi posisi ditambahkan ke dalam embedding menggunakan fungsi sinus dan kosinus.
- **Feed-Forward Network (FFN) atau Multilayer Perceptron (MLP)**. Setelah mekanisme atensi, setiap token diproses secara independen melalui jaringan saraf untuk menyempurnakan fitur-fiturnya secara non-linear.
- **Residual Connections & Layer Normalization**. Ini adalah teknik untuk menjaga stabilitas pelatihan dan mencegah hilangnya sinyal saat data mengalir melalui banyak lapisan (hingga 96 lapisan pada GPT-3).
- **Masking**. Pada Decoder, model dilarang melihat kata-kata yang akan datang selama pelatihan agar ia benar-benar belajar memprediksi kata berikutnya.

Dengan perkembangan inovasi di atas, ketika model disuruh memprediksi kata selanjutnya secara terus-menerus pada data skala masif dengan arsitektur Transformer, model tersebut mulai "memahami" tata bahasa, logika, fakta dunia, hingga kemampuan pemrograman. Gabungan antara prediksi sederhana dan mekanisme *Attention* yang memahami konteks inilah yang melahirkan kecerdasan buatan sintetik masa kini.

---

## 3. Klasifikasi Tugas NLP dan Metrik Evaluasi
Mari sedikit mereview materi yang telah kita pelajari di pertemuan sebelumnya. 

Transformer digunakan untuk berbagai jenis tugas yang dapat dikategorikan menjadi tiga kelompok:
1.	**Classification**. *Classification* adalah memprediksi satu label (misal: analisis sentimen positif/negatif) dengan metrik **Akurasi, Precision, Recall, F1 Score**.
2.	**Multi-classification**. *Multi-classification* adalah memprediksi banyak label (misal: *Named Entity Recognition* untuk mendeteksi lokasi atau waktu).
3.	**Generation**. *Generation* adalah menghasilkan teks baru (misal: penerjemahan, ringkasan, chatbot) dengan metrik:
- BLUE (Bilingual Evaluation Understudy) untuk terjemahan
- ROUGE untuk ringkasan
- Perplexity untuk mengukur seberapa "bingung" model terhadap outputnya sendiri.

---

## 4. KAPASITAS MODEL: PARAMETER VS TINGKAT KECERDASAN

### A. Apa itu Parameter dalam LLM?

Di minggu ke-5, kita telah belajar tentang *Artificial Neural Network* (ANN) yang memiliki *weights* (bobot) dan *biases*. Dalam LLM, bobot dan bias inilah yang disebut sebagai **Parameter**. 

Kita bisa membayangkan parameter sebagai **"sinapsis" atau koneksi saraf** di dalam otak buatan model tersebut. Semakin banyak parameter, semakin banyak pola, bahasa, dan pengetahuan yang bisa "diingat" dan dihubungkan oleh AI.
*   **GPT-1 (2018):** ~117 Juta parameter (Mulai bisa menulis paragraf singkat).
*   **GPT-3 (2020):** ~175 Miliar parameter (Bisa coding, menulis essay panjang, dan lulus ujian manusia).

### B. Scaling Laws: Semakin Besar Pasti Semakin Pintar?

Di dunia AI saat ini, terdapat sebuah hukum yang diyakini perusahaan teknologi besar, yaitu **Scaling Laws**. Hukum ini secara sederhana menyatakan bahwa:
> *Kecerdasan sebuah model AI akan terus meningkat secara terprediksi seiring dengan penambahan Ukuran Model (Parameter), Jumlah Data Pelatihan, dan Daya Komputasi (GPU).*

Inilah alasan mengapa OpenAI, Google, dan Meta menghabiskan miliaran dolar untuk membangun superkomputer raksasa. Namun, ini memunculkan pertanyaan: Apakah untuk mencapai *Artificial General Intelligence (AGI)* kita hanya perlu terus menambah jumlah GPU dan ukuran parameter? Ataukah kita akan segera menabrak batas, mengingat kita mulai kehabisan teks berkualitas tinggi buatan manusia di internet untuk dijadikan data latih?

### C. Debat Kelas: Data Berkualitas vs Komputasi Raksasa

   > *"Model Llama-3 dari Meta yang berukuran 'hanya' 8 Biliar parameter terbukti mampu mengalahkan performa model-model AI generasi sebelumnya yang berukuran raksasa, lebih dari 100 Biliar parameter. Menurut kalian, kenapa model yang otaknya lebih kecil bisa lebih cerdas?"*

Kunci dari model-model modern adalah **Kualitas Data**. Konsep *Chinchilla Scaling Laws* membuktikan bahwa melatih model kecil dengan data yang sangat bersih, terstruktur, dan berkualitas tinggi jauh lebih efektif daripada menjejalkan data "sampah" ke model raksasa. Selain itu, teknik penyelarasan seperti **RLHF (Reinforcement Learning from Human Feedback)** dan instruksi *fine-tuning* memainkan peran sangat besar untuk membuat AI yang "sopan" dan berguna, bukan sekadar beo mekanik.

---

💡 **Takeaway Minggu Ini:**
Kecerdasan LLM saat ini bukanlah sihir, melainkan kemampuan matematis raksasa untuk memprediksi kata selanjutnya secara akurat, yang dimungkinkan karena model tersebut tahu kata mana yang harus diberikan "Perhatian" (*Attention*) secara paralel.
