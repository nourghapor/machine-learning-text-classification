import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Metin temizleme fonksiyonu: küçük harf, noktalama ve rakam kaldırma, lemmatizasyon
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def confusion_matrix_self(y_true, y_pred, class_names, model_name="Model"):
    # Karmaşıklık matrisi çizimi ve yazdırılması
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print(f"{model_name} Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                linecolor='black', linewidth=0.5,
                annot_kws={"size": 12})
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title(f'{model_name} Karmaşıklık Matrisi')
    plt.show() 

def run_model(df, categories, text_column):
    # Verisetini kategorilere göre filtrele
    df = df[df['category'].isin(categories)].copy()
    # Temizlenmiş metin sütunu oluştur
    df['cleaned_text'] = df[text_column].apply(clean_text)

    # TF-IDF ile metni vektörleştir
    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(df['cleaned_text'])
    y = df['category']

    # Eğitim ve test verisine ayır
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # SMOTE ile eğitim verisini dengele
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    # Naive Bayes modelini eğit ve değerlendir
    nbModel = MultinomialNB()
    nbModel.fit(x_train_res, y_train_res)
    nb_pred = nbModel.predict(x_test)
    print("Naive Bayes results")
    print("Accuracy:", accuracy_score(y_test, nb_pred))
    confusion_matrix_self(y_test, nb_pred, class_names=categories, model_name="Naive Bayes")
    print(classification_report(y_test, nb_pred))

    # Logistic Regression modelini eğit ve değerlendir
    lrModel = LogisticRegression(max_iter=1000, class_weight='balanced')
    lrModel.fit(x_train_res, y_train_res)
    lr_pred = lrModel.predict(x_test)
    print("Logistic Regression results")
    print("Accuracy:", accuracy_score(y_test, lr_pred))
    confusion_matrix_self(y_test, lr_pred, class_names=categories, model_name="Logistic Regression")
    print(classification_report(y_test, lr_pred))

    return vectorizer, lrModel

# ----------- Veri Yükleme ve Hazırlık -------------

# Kaggle veri setleri
df = pd.read_json('E:\\news_text_classification\\news-category-dataset\\News_Category_Dataset_v3.json', lines=True)
df1 = pd.read_csv('E:\\news_text_classification\\news-category-dataset\\train.csv', header=None, names=['category', 'headline', 'description'])

# df1 kategorileri sayısaldan stringe çevir
categories1_map = {
    '1': 'POLITICS',
    '2': 'SPORTS',
    '3': 'BUSINESS'
}
df1['category'] = df1['category'].astype(str).map(categories1_map)

# df1'de headline ve description birleşimi ile 'text' sütunu oluştur
df1['text'] = df1['headline'].fillna('') + ' ' + df1['description'].fillna('')

# df için de 'text' sütunu oluştur (headline tek başına)
df['text'] = df['headline'].fillna('')

# Kategori listesi - ortak kullanacağımız
categories = ['BUSINESS', 'SPORTS', 'ENTERTAINMENT', 'POLITICS']

# ------------ Model Eğitimleri -------------

print("DF News_Category_Dataset_v3")
vectorizer, model = run_model(df, categories, 'text')

print("DF1 train.csv")
categories_df1 = list(set(categories1_map.values()))
vectorizer, model = run_model(df1, categories_df1, 'text')

print("Both datasets combined")

# Sadece gerekli sütunları alarak birleştir (kategori ve text)
combined_df = pd.concat([df[['category','text']], df1[['category','text']]], ignore_index=True)

# Kategorileri filtrele (entertainment df1'de yok ama df'de var)
combined_df = combined_df[combined_df['category'].isin(categories)]

# Birleşik veri seti ile model eğitimi
vectorizer, model = run_model(combined_df, categories, 'text')

# -------------- Kullanıcıdan Metin Tahmini -------------

while True:
    try:
        print("\n Değer giriniz (to exit press 'q' or 'Q'):")
        text = input()
        if text.lower() == 'q':
            print("Program sonlandırıldı.")
            break
        if not text.strip():
            print("Lütfen boş olmayan bir metin giriniz.")
            continue
        text_clean = clean_text(text)
        text_vectorized = vectorizer.transform([text_clean])
        prediction = model.predict(text_vectorized)
        print("Tahmini kategori::", prediction[0])
    except Exception as e:
        print(f"Bir hata oluştu: {e}. Lütfen tekrar deneyiniz.")
