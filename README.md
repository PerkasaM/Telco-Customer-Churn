# Prediksi Churn Pelanggan pada Perusahaan Xfinity Menggunakan Machine Learning
## **Latar Belakang**

Comcast Corp bergerak dalam penyediaan layanan video, Internet, dan telepon. Ini beroperasi melalui segmen berikut: Komunikasi Kabel, Media, Studio, Taman Hiburan, dan Langit. Segmen Komunikasi Kabel terdiri dari Comcast Cable, yang merupakan penyedia layanan broadband, video, suara, nirkabel, dan lainnya kepada pelanggan perumahan di Amerika Serikat dengan merek Xfinity. Xfinity adalah segmen bisnis telekomunikasi Amerika dan divisi dari Comcast Corporation yang digunakan untuk memasarkan televisi kabel konsumen , internet , telepon , dan layanan nirkabel yang disediakan oleh perusahaan. Merek ini pertama kali diperkenalkan pada tahun 2010; sebelumnya, layanan ini dipasarkan terutama dengan nama Comcast.

Segmen Media terdiri dari platform televisi dan streaming NBCUniversal, termasuk jaringan kabel nasional, regional, dan internasional. Segmen Studio berfokus pada operasi produksi dan distribusi studio film dan televisi NBCUniversal. Segmen Taman Hiburan mengoperasikan taman hiburan Universal di Orlando, Florida, Hollywood, California, Osaka, Jepang, dan Beijing, Tiongkok. Segmen Sky menyediakan operasi Sky, salah satu perusahaan hiburan Eropa, yang terutama mencakup bisnis langsung ke konsumen, menyediakan layanan video, broadband, suara dan telepon nirkabel, dan bisnis konten, mengoperasikan jaringan hiburan, jaringan siaran Sky News , dan jaringan Sky Sports. Perusahaan ini didirikan pada tahun 1963 dan berkantor pusat di Philadelphia, PA.



Reference :

https://en.wikipedia.org/wiki/Xfinity

https://edition.cnn.com/markets/stocks/CMCSA
## **Pernyataan Masalah**


Pada saat ini persaingan semakin ketat pada bidang telekomunikasi. Saat agustus 2023 kapitalisasi pasar masih dipimpin oleh Comcast (Xfinity) dilanjutkan dengan Netflix, Waltdisney, Sony dan Activision Blizzard. perbedaan antara perusahaan dengan Netflix sangat tipis sekali. Namun hal ini menjadi peringatan untuk perusahaan untuk mulai melakukan pencegahan agar dapat memimpin pasar. Langkah yang akan diambil perusahaan adalah untuk mulai mencoba menganalisa Churn pelanggan dan memprediksi pelanggan yang sudah dimiliki. Kali ini tim Data Scientist pada perusahaan Xfinity untuk membuat menganalisis dan mengidentifikasi pola.

Churn pelanggan sering disebut sebagai attrisi pelanggan, atau defeksi pelanggan yang merupakan tingkat dimana pelanggan hilang. Perusahaan telekomunikasi sering menggunakan churn pelanggan sebagai metrik bisnis utama untuk memprediksi jumlah pelanggan yang akan meninggalkan penyedia layanan telekomunikasi. Churn sangat signifikan dalam industri telekomunikasi karena secara langsung memengaruhi daya saing penyedia layanan.

Faktanya mendapatkan pelanggan baru pada memerlukan **biaya 7x yang lebih mahal** di banding mempertahankan pelanggan yang lama.
Churn pelanggan adalah kerugian langsung dalam hal pendapatan bagi perusahaan. Jika informasi tentang perkembangan churn pelanggan diketahui jauh sebelumnya, maka langkah-langkah yang tepat dapat diambil, dan layanan yang lebih baik dapat diberikan kepada pelanggan tersebut. Diperhatikan bahwa pelanggan jangka panjang menambah lebih banyak pendapatan bagi perusahaan karena mereka tidak terlalu responsif terhadap perubahan kecil. Salah satu masalah yang paling menantang dan kritis yang dihadapi oleh industri telekomunikasi saat ini adalah manajemen pelanggan yang churn.


references :

https://www.sciencedirect.com/science/article/abs/pii/S0148296318301231

https://bstrategyhub.com/comcast-competitors-and-alternatives/

https://databoks.katadata.co.id/datapublish/2023/08/30/salip-netflix-comcast-jadi-perusahaan-hiburan-terbesar-dunia-pada-agustus-2023

## **Analisa Tim Data Science**

Dalam konteks case churn, FP, FN, TP, dan TN mengacu pada kondisi yang mungkin terjadi dalam proses prediksi churn pelanggan. Berikut adalah arti dari masing-masing istilah tersebut:

1. **True Positive (TP):**
   - TP terjadi ketika model prediksi berhasil mengidentifikasi pelanggan yang benar-benar cenderung untuk beralih (churn).
   - Dalam konteks ini, TP menggambarkan situasi di mana model berhasil memprediksi pelanggan yang benar-benar melakukan churn, sehingga perusahaan dapat mengambil langkah-langkah retensi yang tepat untuk mempertahankan mereka.

2. **True Negative (TN):**
   - TN terjadi ketika model prediksi berhasil mengidentifikasi pelanggan yang sebenarnya tidak cenderung untuk beralih (tidak churn).
   - Dalam konteks ini, TN menggambarkan situasi di mana model berhasil memprediksi pelanggan yang sebenarnya tetap loyal dan tidak melakukan churn, sehingga perusahaan tidak perlu mengalokasikan sumber daya untuk upaya retensi pada pelanggan tersebut.

3. **False Positive (FP):**
   - FP terjadi ketika model prediksi salah mengidentifikasi pelanggan yang sebenarnya tidak cenderung untuk beralih sebagai pelanggan yang cenderung untuk beralih.
   - Dalam konteks ini, FP menggambarkan situasi di mana model salah memprediksi bahwa pelanggan akan melakukan churn, sehingga perusahaan mungkin mengalokasikan sumber daya untuk upaya retensi pada pelanggan yang sebenarnya tidak memerlukannya.

4. **False Negative (FN):**
   - FN terjadi ketika model prediksi gagal mengidentifikasi pelanggan yang sebenarnya cenderung untuk beralih sebagai pelanggan yang tidak cenderung untuk beralih.
   - Dalam konteks ini, FN menggambarkan situasi di mana model gagal memprediksi bahwa pelanggan akan melakukan churn, sehingga perusahaan mungkin kehilangan kesempatan untuk mengambil langkah-langkah retensi yang diperlukan untuk mempertahankan pelanggan tersebut.


Berdasarkan analisa dan pernyataan masalah yang sudah dijelaskan, bahwa untuk mencari pelanggan baru memerlukan **biaya 7x lebih besar** di bandingkan dengan memberikan fasilitas terhadap pelanggan. Fokus pada tim Data Scientist adalah pada FN untuk **meminimalkan pelanggan yang akan churn**, maka metric yang kami gunakan adalah **Recall**. 

# **Biaya yang dikeluarkan kami mengasumsikan:**
- Retain Cost
   Retain cost adalah total biaya untuk mempertahankan seorang pelanggan, termasuk semua biaya yang dikeluarkan selama proses pemasaran dan penjualan hingga pada titik transaksi. Dengan mengasumsikan ***per pelanggan $10***.

- Acquisition Cost 
   Acquisition Cost adalah biaya yang dibutuhkan untuk menyakinkan calon pelanggan agar membeli suatu produk atau layanan. Dengan mengasumsikan ***per pelanggan baru $70*** 


refrences :

https://www.totango.com/glossary/customer-retention-cost-crc

### Data


| No. | Columns            | Value                |
|-----|---------------------|----------------------|
| 1   | Dependents          | Apakah pelanggan memiliki tanggungan atau tidak.|
| 2   | Tenure      | Jumlah bulan pelanggan telah berlangganan dengan perusahaan.|
| 3   | OnlineSecurity     | Apakah pelanggan memiliki keamanan online atau tidak.|
| 4   | OnlineBackup     | Apakah pelanggan memiliki cadangan online atau tidak.|
| 5   | InternetService    | Apakah pelanggan berlangganan layanan internet atau tidak.|
| 6   | DeviceProtection | Apakah pelanggan memiliki perlindungan perangkat atau tidak.|
| 7   | TechSupport     | Apakah pelanggan memiliki dukungan teknis atau tidak.|
| 8   | Contract        | Jenis kontrak berdasarkan durasinya.|
| 9   | PaperlessBilling| Apakah tagihan dikeluarkan dalam bentuk tanpa kertas atau tidak.|
| 10  | MounthlyCharges     | Jumlah biaya layanan per bulan dalam mata uang tertentu.|
| 11  | Churn               | Apakah pelanggan churn (berhenti berlangganan) atau tidak.|



**Kesimpulan**

1. Model yang digunakan adalah logistic regression karena hasil dari cross validasi mean 0.516505 lebih besar dibanding model yang lain, dan standar deviasinya 0.030946 kecil dibandingnya mean. Selain itu dengan menggunakan logistic regression lebih mudah dalam melakukan prediksi.

2. Perbedaan model yang belum dilakukan resampling dengan yang sudah di resampling cukup meningkat lebih baik 0.333 dari sebelum resampling 0.520 menjadi 0.853 dengan menggunakan random oversamapling.

3. Dampak dalam bisnis jika di asumsikan bahwa akan ada 1000 orang pelanggan.
   Biaya retain cost = $10
   Biaya acquistion cost = $70

    - jika menggunakan tidak menggunakan machine learning dengan asumsi optimis 0,5 maka :
     
         - yang di prediksi oleh ML : 1000 x 0,50 = 500
         - yang tidak terprediksi ML : 1000 x 0,50 = 500
         - Retain cost dengan ML = 500 x 10 = $5000
         - Acquisition cost dengan ML = 500 x 70 = $35000
         - total = 5000 + 35000 = $40.000

    - jika menggunakan machine learning dengan best model dan menggunakan random undersampling 0,82 maka :
     
         - yang di prediksi oleh ML : 1000 x 0,82 = 820
         - yang tidak terprediksi ML : 1000 x 0,18 = 180
         - Retain cost dengan ML = 820 x 10 = $8200
         - Acquisition cost dengan ML = 180 x 70 = $12600
         - total = 8200 + 12600 = $20.800

    Selisih antara menggunakan biaya menggunakan machine learning dan tidak
    
    40.000 - 20.000 = 20.000

    dengan menggunakan machine learning biaya untuk mengatasi churn dapat diturunkan 50%

## **Rekomendasi**

Model

- dapat menambahkan model yang lain untuk melakukan hyperparameter tuning seperti SVM
- pada tahap preprocessing dapat dicoba pada scaler dengan menggunakan min max scaler
- pada tahap hyperparameter tuning pada ensamble dapat menambahkan Adaboost
- perlu ditambahkan feature agar prediksi yang dilakukan machine learning maksimal


Bisnis

- Customer yang Contract perbulan memiliki potensi untuk churn tinggi sehinggi perlu diperhatikan oleh perusahaan agar tidak churn dengan memberi layanan seperti diskon agar dapat customer menjadi tidak churn

- Customer yang tidak menggunakan Paperless billing perlu diperhatikan oleh perusahaan agar tidak terjadi churn.

- Customer yang menggunakan layanan internet fiber optic perlu diperhatikan oleh perusahaan agar tidak terjadi churn