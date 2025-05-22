<<<<<<< HEAD
# News Text Classification with Machine Learning

## Proje Hakkında
Bu proje, iki farklı kamuya açık İngilizce haber veri seti kullanarak haber metinlerinin otomatik olarak kategorilere sınıflandırılmasını amaçlamaktadır.  
Projede, temel makine öğrenmesi modelleri olan **Multinomial Naive Bayes** ve **Logistic Regression** kullanılmıştır.  
Veri ön işleme adımları, model eğitimi, performans değerlendirmesi ve interaktif metin tahmini içermektedir.

---

## Kullanılan Veri Setleri
- **News Category Dataset**  
  Kaynak: [Kaggle - News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)  
  Yaklaşık 200.000 haber başlığı içerir. Kategoriler: BUSINESS, SPORTS, ENTERTAINMENT, POLITICS  

- **AG News Classification Dataset**  
  Kaynak: [Kaggle - AG News Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  
  Yaklaşık 120.000 haber başlığı ve açıklaması içerir. Kategoriler: POLITICS, SPORTS, BUSINESS, ENTERTAINMENT

---

## Proje Özellikleri
- Veri temizleme: Küçük harfe çevirme, noktalama ve sayıları temizleme, kelime köklerine indirgeme (lemmatizasyon)  
- İki farklı makine öğrenmesi modelinin uygulanması ve karşılaştırılması  
- Veri setlerindeki kategori dengesizliğini gidermek için SMOTE ile oversampling  
- Model performansını ölçmek için doğruluk, sınıflandırma raporu ve karmaşıklık matrisi görselleştirmeleri  
- Kullanıcıdan alınan metinler için gerçek zamanlı kategori tahmini  

---

## Kurulum ve Kullanım

### Gereksinimler
- Python 3.7 veya üstü  
- Gerekli kütüphaneler:  
  ```bash
  pip install -r requirements.txt
=======
# machine-learning-text-classification
A Python project for news text classification using machine learning models (Naive Bayes and Logistic Regression) on public news datasets. Includes data preprocessing, model training, evaluation, and interactive text prediction.
>>>>>>> bc692569205427088bfcb8e8e4a1226c047feda2
