# Tradisional Home classification With VGG16 & MobileNetv2

## Deskripsi Project
Project ini bertujuan untuk mengklasifikasikan gambar rumah adat Indonesia berdasarkan desain dan ciri khasnya. Indonesia memiliki keanekaragaman budaya yang tercermin dalam rumah adat yang berbeda di setiap daerah. Dalam proyek ini, kami mengembangkan model untuk mengenali berbagai jenis rumah adat dari gambar dengan tujuan untuk mempermudah pengenalan dan pelestarian budaya Indonesia.

## Dataset
Dataset yang digunakan dalam proyek ini dapat diakses di [Kaggle: Rumah Adat Indonesia](https://www.kaggle.com/datasets/rariffirmansah/rumah-adat). Dataset ini berisi gambar rumah adat terdiri dari 3919 rumah adat, yang digunakan untuk melatih model deep learning dalam tugas klasifikasi. terdiri dari kategori :

![Gambar1](https://github.com/FebbyNurIdhananto12/RumahAdat_classification/blob/main/gambar/dataset.png)

### 1. Persyaratan Sistem
Pastikan Anda telah menginstal Python versi 3.10 atau lebih baru.

### 2. Masuk direktori
```bash
cd repository
```

### 3. Instalasi 
Setelah menyiapkan lingkungan Python, instal yang diperlukan dengan menjalankan perintah berikut:
```bash
pip install -r requirements.txt
```

### 4. run app
```bash
pdm run mulai
```

## Pustaka 
streamlit
numpy
tensorflow
PIL

## Deskrisi
VGG16 adalah pilihan terbaik jika sumber daya komputasi tersedia dan membutuhkan model dengan akurasi tinggi untuk tugas pengenalan gambar besar.
MobileNetV2 sangat cocok untuk aplikasi yang membutuhkan efisiensi komputasi tinggi dan kecepatan inferensi, seperti pada perangkat mobile atau aplikasi embedded.

## Hasil
### VGG16
#### Val dan loss
![Gambar2](https://github.com/FebbyNurIdhananto12/RumahAdat_classification/blob/main/gambar/akurasi_val_vgg16.png) ![Gambar3](https://github.com/FebbyNurIdhananto12/RumahAdat_classification/blob/main/gambar/akurasi_loss_vgg16.png)

#### Classification Report
![Gambar4](https://github.com/FebbyNurIdhananto12/RumahAdat_classification/blob/main/gambar/clasification_report%20_vgg.jpg)

#### Confussion Matrix
![Gambar5](https://github.com/FebbyNurIdhananto12/RumahAdat_classification/blob/main/gambar/Confusionmatrix_vgg.png)

### VGG16
#### Val dan loss
![Gambar6](https://github.com/FebbyNurIdhananto12/RumahAdat_classification/blob/main/gambar/akurasi%20val%20mobilenet.png)
![Gambar7](https://github.com/FebbyNurIdhananto12/RumahAdat_classification/blob/main/gambar/akurasi_loss_mobilenet.png)

#### Classification Report
![Gambar8](https://github.com/FebbyNurIdhananto12/RumahAdat_classification/blob/main/gambar/clasification_report%20.jpg)

#### Confussion Matrix
![Gambar9](https://github.com/FebbyNurIdhananto12/RumahAdat_classification/blob/main/gambar/Confusionmatrix_mobilenet.png)
