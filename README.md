# Laporan Proyek Machine Learning - Razif Zulvikar Hatuwe

## A. Domain Proyek 
Latar Belakang 

Sistem rekomendasi semakin penting dalam membantu pengguna menemukan produk atau layanan yang relevan di tengah banyaknya pilihan. Proyek ini bertujuhan untuk mengembangkan sistem rekomendasi berbasis collaborative filtering yang dapat memberikan rekomendasi produk secara akurat kepada pengguna Amazon. Penyelesaian proyek ini penting karena mampu meningkatkan pengalaman pengguna dengan memberikan rekomendasi produk yang sesuai, sehingga mendorong loyalitas pengguna dan meningkatkan penjualan.

## B. Business Understanding

**Problem Statements** 

Sistem rekomendasi pada platform Amazon perlu memberikan saran produk yang relevan dan sesuai dengan preferensi pengguna, tetapi sulit untuk memilih model yang menghasilkan akurasi tinggi dan efisiensi dalam skala besar. Oleh karena itu, proyek ini akan membandingkan 2 (dua) algoritma utama (KNN dan SVD) untuk menemukan pendekatan terbaik dalam memberikan rekomendasi produk.

**Goals**

Tujuan dari proyek ini adalah untuk mengembangkan model rekomendasi produk berbasis collaborative filtering yang efektif dan efisien untuk meningkatkan relevansi rekomendasi. Sistem ini diharapkan mampu memberikan saran yang tepat bagi pengguna, sehingga dapat mendorong keterlibatan dan kepuasan pengguna.

**Solution Statements**

Untuk mencapai tujuan proyek, akan diterapkan dua pendekatan utama: model K-Nearest Neighbors (KNN) dan Singular Value Decomposition (SVD). Model KNN akan digunakan dalam dua varian: User-to-user Collaborative Filtering dan Product-to-product Collaborative Filtering. Model SVD, yang terkenal karena kemampuannya dalam mempresentasikan hubungan antar fitur dalam ruang vektor, akan digunakan sebagai pendekatan alternatif untuk membandingkan hasil dan peforma.

## C. Data Understanding
Dataset yang digunakan dalam proyek ini merupakan data interaksi pengguna terhadap produk di platform Amazon, yang mengandung informasi mengenai rating yang diberikan oleh pengguna.

1. **Sumber Data**:
Dataset ini diperoleh dari kaggle, yang menyediakan data interkasi pengguna terhadap produk. Sumber dataset ini dapat di akses melalui tautan link berikut [https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews].

2. **Jumlah Data (Baris dan Kolom)**:
Dataset ini memiliki total **7.824.481 baris** dan **4 kolom**. Setiap baris mewakili satu interaksi antara pengguna dengan produk tertentu.

3. **Kondisi Data**: 
Berdasarkan analisis awal : 
    - **Missing Values**: Tidak ditemukan adanya missing value pada seluruh kolom (user_id, product_id, Rating, dan timestamp).
    - **Duplicate Values**: Tidak ada baris yang terduplikasi, sehingga seluruh data dianggap unik.
    - **Kondisi Umum**: Data ini bersih dari missing values dan duplikasi, yang menjadi keuntungan dalam tahap data preparation karena meminimalkan perlunya data cleansing lebih lanjut.

4. **Struktur dataset**:

| Nama Tabel | Info Table                                                                             |
|------------|----------------------------------------------------------------------------------------|
| user_id    | Setiap pengguna diidentifikasi dengan ID unik.                                         |
| product_id | Setiap produk diidentifikasi dengan ID unik.                                           |
| rating     | Skor yang diberikan oleh pengguna terhadap produk, dengan skala antara 1.0 hingga 5.0. |
| timestamp  | waktu saat rating diberikan, direpresentasikan dengan format Unix timestamp.           |

## D. Data Preparation
Pada bagian data preparation ini, melakukan beberapa langkah untuk menyiapkan data agar dapat di gunakan oleh pustaka Surprise: 

1. **Sampling Data** (```data = data.sample(n=10000, random_state=42)```)
Langkah ini mengambil sampel sebanyak 10.000 data dari dataset asli. Sampling digunakan untuk mengurangi ukuran data, sehingga pemrosesan lebih cepat dan efisien. Dengan menggunakan parameter ```random_state=42```, sampling ini menghasilkan data yang konsisten dan dapat direproduksi.

2. **Mengidentifikasikan Reader** (```reader = Reader(rating_scale=(1, 5))```)
Langkah ini membuat objek Reader yang menetapkan rentang skala rating, yaitu dari 1 hingga 5. Ini penting karena Surprise memerlukan definisi rentang skala rating agar dapat membaca data dengan benar. Setiap nilai rating harus berada dalam rentang ini untuk menghindari kesalahan dalam perhitungan dan analisis model.

3. **Mempersiapkan Dataset** (```surprise_data = Dataset.load_from_df(data[['user_id', 'product_id', 'Rating']], reader)```) 
Langkah ini memuat data dari DataFrame ke dalam format Dataset yang dikenali oleh Surprise. Kolom ```user_id```, ```product_id```, dan ```Rating``` yang dipilih karena ketiganya adalah komponen inti dalam collaborative filtering: ```user_id``` mengidentifikasi pengguna, ```product_id``` mengidentifikasi produk, dan Rating menunjukkan penilaian yang diberikan oleh pengguna untuk produk tersebut. Langkah ini penting untuk menyiapkan data dalam format yang kompatibel dengan Surprise dan untuk memfasilitasi pembuatan dataset yang dapat digunakan dalam melatih dan menguji model.

4. **Membagi Data Menjadi Trainset dan Testset** (```trainset, testset = train_test_split(surprise_data, test_size=0.2)```)
Data kemudian dibagi menjadi trainset dan testset dengan porsi 80% data untuk pelatihan dan 20% untuk pengujian. Membagi data ini penting untuk mengevaluasi performa model. Dengan melakukan pembagian, Anda dapat menguji model pada data yang tidak pernah dilihat selama pelatihan, sehingga menghasilkan evaluasi yang lebih obyektif terhadap akurasi model.

## E. modeling
Pada bagian modeling terdapat dua pendeakatan algoritma yaitu model KNN dan model SVD. 
1. **Model KNN (K-Nearest Neighbors)**
KNN adalah algoritma yang populer dalam collaborative filtering. Pada proyek ini, KNN diterapkan dalam dua varian: User-to-User Collaborative Filtering dan Product-to-Product Collaborative Filtering.

a. **Parameter dalam Model KNN** 
| Param       | Keterangan                                                                                                                                                                                                                                                                                                                                                                                             |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```sim_options``` | Parameter ini menentukan metode pengukuran kesamaan yang digunakan dalam model                                                                                                                                                                                                                                                                                                                         |
| ```name```        | Menyebut jenis matrik kesamaan yang akan digunakan. Dalam hal ini, ```cosine``` digunakan untuk mengukur kesamaan sudut antara vektor pengguna atau produk.                                                                                                                                                                                                                                                  |
| ```user_based```  | Parameter boolean yang menentukan jenis pendekatan collaborative filtering. Jika ```True```, model menggunakan pendekatan User-to-User Collaborative Filtering, dimana kesamaan antara pengguna dihitung berdasarkan pola rating mereka. Jika ```False```, model menggunakan Product-to-Product Collaborative Filtering, dimana kesamaan antar produk dihitung berdasarkan pengguna yang memberikan rating serupa. |    

b. **Cara Kerja Model KNN**
    1. **Penghitungan Kesamaan**: Model menghitung kesamaan antara pengugna atau produk berdasarkan metrik kesamaan yang ditentukan (```cosine```). Misalnya dalam pendekatan User-to-User Collaborative Filtering, Kesamaan antara pengguna dihitung berdasarkan pola rating mereka terhadap berbagai produk. Begitu juga, pada pendekatan Product-to-Product Collaborative Filtering, kesamaan antara produk dihitung berdasarkan pengguna yang telah memberikan rating pada produk tersebut.
    2. **Identifikasi Tetangga Terdekat**: Setelah kesamaan dihitung, model akan menentukan sejumlah tetangga terdekat (```K``` Neighbors). Tetangga terdekat ini adalah pengguna atau produk yang memiliki tingkat kesamaan tertinggi.
    3. **Prediksi Rating**: Berdasarkan tetangga terdekat yang telah diidentifikasi, model akan memprediksi rating untuk produk tertentu dengan mengkombinasikan rating dari tetangga tersebut, misalnya dengan mengambil rata-rata rating mereka.

2. **Model SVD (Singular Value Decomposition)**
SVD merupakan teknik dekomposisi matrik yang digunakan dalam collabirative filtering untuk mempresentasikan relasi antara pengguna dan produk dalam ruang yang lebih terkompresi. SVD berguna karena dapat mengurangi dimensionalitas data dan menangkap pola laten yang tidak langsung terlihat pada data mentah.

a. Parameter dalam Model SVD
| Param     | Keterangan                                                                                                                                                                                                                                                                                             |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```n_factors``` | Menentukan jumlah faktor laten atau dimensi yang akan digunakan dalam dekomposisi matrik. Secara default, nilai ini biasanya sekitar 100. Semakin tinggi nilai ```n_factors```, semakin kompleks dan detail hubungan yang dapat direpresentasikan, tetapi juga memerlukan komputasi yang lebih besar.        |
| ```n_epochs```  | Jumlah iterasi atau epoch yang akan dijalankan oleh algoritma. Setiap epoch berarti model melakukan satu putara untuk mengupdate faktor lane. Semakin tinggi jumlah epoch, model akan semakin terlatin, Tetapi juga membutuhkan waktu komputasi yang lebih lama.                                       |
| ```lr_all```    | Learning rate atau laju pembelajaran untuk semua parameter. Nilai ini mengontrol besarnya langkah yang diambil dalam setiap iterasi untuk mengupdate faktor laten pengguna dan produk. Nilai yang lebih besar dapat mempercepat konvergensi tetapi berisiko mengalami overfitting jika terlalu beasr.  |
| ```reg_all```   | Parameter regularisasi yang digunakan untuk mencegah overfitting dengan menambahkan penalti pada model ketika parameter menjadi terlalu beasr.                                                                                                                                                         |

b. Cara Kerja Model SVD 
    1. **Dekomposisi Matrik**: Model SVD melakukan dekomposisi pada matriks rating menjadi tiga matriks: matriks pengguna (```U```), matriks diagonal dari faktor laten (```Σ```), dan matriks produk (```V```). Ini membantu merepresentasikan hubungan antara pengguna dan produk dalam ruang vektor dengan lebih sedikit dimensi.
    2. **Pembelajaran Faktor Laten**: Model mengupdate faktor laten pengguna dan produk melalui teknik optimasi (seperti stochastic gradient descent) dengan mempertimbangkan parameter ```n_factors```, ```lr_all```, dan ```reg_all```. Dalam proses ini, SVD belajar untuk memprediksi rating yang diberikan pengguna terhadap produk tertentu berdasarkan faktor laten.
    3. **Prediksi Rating**: Setelah faktor laten diperoleh, model dapat memprediksi rating pengguna terhadap produk dengan cara mengalikan faktor laten pengguna dan produk. Ini menghasilkan nilai prediksi yang merepresentasikan perkiraan preferensi pengguna.

3. **Hasil Top N Recommendations**

a. Top 5 KNN Model Recommendations For User A2WNBOD3WNDNKT
| Rank | Product ID | Predicted Rating |
|------|------------|------------------|
| 1    |  B000M9ISQ2             |   4.00                           |
| 2    |  B002L6HE9G             |   4.00                           |
| 3    |  B007MXGG5Q             |   4.00                           |
| 4    |  B008HOEDYU             |   4.00                           |
| 5    |  B0069R7TAM             |   4.00                           |

b. Top 5 SVD Model Recommendations For User A2WNBOD3WNDNKT
| Rank | Product ID | Predicted Rating |
|------|------------|------------------|
| 1    |  B003ES5ZUU             |   4.58                           |
| 2    |  B0002L5R78             |   4.55                           |
| 3    |  B0019EHU8G             |   4.54                           |
| 4    |  B000S5Q9CA             |   4.53                           |
| 5    |  B00622AG6S             |   4.50                           |

## F. Evaluasi 
**Metrik Evaluasi**

1. Mean Absolute Error (MAE): Mengukur rata-rata kesalahan absolut antara nilai aktual dan prediksi. Semakin rendah Nilai MAE menunjukkan akurasi yang lebih baik.
Rumus RMSE:

RMSE = sqrt( (1/n) * Σ(yᵢ - ŷᵢ)² )

Di mana:
- `n` adalah jumlah data dalam test set,
- `yᵢ` adalah rating aktual pengguna pada item ke-`i`,
- `ŷᵢ` adalah rating prediksi model pada item ke-`i`.

2. Root Mean Squared Error (RMSE): Mengukur rata-rata kesalahan kuadrat antara nilai aktual dan prediksi. Semakin kecil nilai RMSE, semakin tinggi akurasi model.
Rumus MAE:

MAE = (1/n) * Σ|yᵢ - ŷᵢ|

Di mana:
- `n` adalah jumlah data dalam test set,
- `yᵢ` adalah rating aktual pengguna pada item ke-`i`,
- `ŷᵢ` adalah rating prediksi model pada item ke-`i`.


**Hasil Evaluasi**
1. KNN Model
    - RMSE: 1.3477
    - MAE: 1.0700
2. SVD Model
    - RMSE: 1.3408
    - MAE: 1.0659

Dari hasil evaluasi, dapat dilihat bahwa model SVD memiliki nilai MAE dan RMSE yang lebih rendah di bandingkan model KNN. Ini menunjukkan bawah model SVD memiliki peform ayang lebih baik dalam memprediksi rating produk oleh pengguna karena memiliki error prediksi yang lebih kecil.

**Kesimpulan** 

1.  Problem Statements dan Goals 
a. Proyek ini bertujuan untuk membangun sistem rekomendasi yang akurat dengan membandingkan peforma dua algoritma, yaitu KNN dan SVD.

b. Berdasarkan hasil evaluasi, Model SVD menunjukkan peforma yang lebih baik, sehingga memenuhi goal proyek dalam memberikan rekomendasi produk yang lebih relevan dan akurat.

2. Solution Approach 
a. Proyek telah menjalankan dua pendekatan solusi, yaitu menggunakan model KNN dan SVD. Perbandingan peforma kedua pendekatna ini telah dilakukan melalui meterik MAE dan RMSE.

b. Dengan ini, Solution Approach yang dirancang telah berhasil diimpelemtasikan sesuai rencana, menghasilkan model rekomendasi yang dapat dievaluasi dan memberikan hasil yang menjawab tujuan proyek. 






