# Duygu Analizi Uygulaması - Amazon Veri Seti ile VADER Sentiment Analysis

import pandas as pd  # Veri manipülasyonu ve analizi için pandas kütüphanesi
import nltk  # Doğal dil işleme kütüphanesi
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # VADER duygu analizi aracı
from nltk.corpus import stopwords  # Gereksiz kelimeleri (stop words) filtrelemek için
from nltk.tokenize import word_tokenize  # Metni kelimelere ayırmak için tokenizer
from nltk.stem import WordNetLemmatizer  # Kelimeleri kök formlarına dönüştürmek için
from sklearn.metrics import confusion_matrix,classification_report  # Model performans değerlendirme metrikleri

# Gerekli NLTK veri setlerini indirme
nltk.download('vader_lexicon')  # VADER duygu analizi sözlüğü
nltk.download('punkt')  # Cümle ve kelime tokenizasyonu için
nltk.download('stopwords')  # İngilizce stop words listesi
nltk.download('wordnet')  # WordNet lemmatization için
nltk.download('omw-1.4')  # Çok dilli WordNet desteği

# Amazon ürün yorumları veri setini yükleme
df=pd.read_csv("8.bolum/duygu_analizi_amazon_veri_seti.csv")
lemmatizer=WordNetLemmatizer()  # Lemmatizer nesnesini oluşturma

# Metin ön işleme fonksiyonu
def clean_preprocess_data(text):
    tokens=word_tokenize(text)  # Metni kelimelere ayırma
    filtered_tokens=[word for word in tokens if word not in stopwords.words('english')]  # Stop words'leri filtreleme
    lemmatized_tokens=[lemmatizer.lemmatize(word) for word in filtered_tokens]  # Kelimeleri kök formlarına dönüştürme
    processed_text=" ".join(lemmatized_tokens)  # İşlenmiş kelimeleri tekrar birleştirme
    return processed_text

# Veri setindeki tüm yorumlara ön işleme uygulama
df["Review Text2"]=df["reviewText"].apply(clean_preprocess_data)

# VADER duygu analizi aracını başlatma
analyzer=SentimentIntensityAnalyzer()

# Duygu skorunu hesaplayan fonksiyon
def get_sentiments(text):
    score=analyzer.polarity_scores(text)  # Metnin duygu skorlarını hesaplama (pos, neg, neu, compound)
    sentiment=1 if score["pos"]>0 else 0  # Pozitif skor varsa 1, yoksa 0 atama
    return sentiment

# Tüm yorumlara duygu analizi uygulama
df["Sentiment"]=df["Review Text2"].apply(get_sentiments)

# Model performansını değerlendirme
c_matrix=confusion_matrix(df["Positive"],df["Sentiment"])  # Karışıklık matrisi oluşturma
c_report=classification_report(df["Positive"],df["Sentiment"])  # Detaylı sınıflandırma raporu
print(f"Confusion Matrix: {c_matrix}")  # Karışıklık matrisini yazdırma
print(f"Classification Report: {c_report}")  # Sınıflandırma raporunu yazdırma

# KONU AÇIKLAMASI:
# Bu kod, Amazon ürün yorumları üzerinde duygu analizi yapmaktadır.
# VADER (Valence Aware Dictionary and sEntiment Reasoner) algoritması kullanılarak
# müşteri yorumlarının pozitif veya negatif olduğu belirlenmektedir.
# Metin ön işleme adımları ile veriler temizlenir ve daha iyi sonuçlar elde edilir.
# Son olarak, gerçek etiketlerle tahmin edilen etiketler karşılaştırılarak model performansı değerlendirilir.


    






