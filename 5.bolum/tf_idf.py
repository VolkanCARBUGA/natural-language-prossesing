# TF-IDF (Term Frequency - Inverse Document Frequency) analizi için gerekli kütüphaneler
import pandas as pd  # Veri analizi ve DataFrame işlemleri için
import numpy as np   # Sayısal hesaplamalar için

# Scikit-learn kütüphanesinden TF-IDF vektörleyicisi
from sklearn.feature_extraction.text import TfidfVectorizer

# Analiz edilecek örnek belge koleksiyonu
documents = [
    "köpek çok tatlı bir hayvan",           # Belge 1
    "köpekler ve kuşlar çok tatlı hayvanlardır",  # Belge 2
    "inekler  süt verir"                    # Belge 3
]

# TF-IDF vektörleyici nesnesini oluşturma
vectorizer = TfidfVectorizer()

# Belgeleri TF-IDF vektörlerine dönüştürme
# fit_transform: hem modeli eğitir hem de verileri dönüştürür
X = vectorizer.fit_transform(documents)

# Vektörlerdeki feature (kelime) isimlerini alma
feature_names = vectorizer.get_feature_names_out()

# Sparse matrix'i dense array'e çevirme (görselleştirme için)
vector_temsili = X.toarray()
#print(vector_temsili)  # Vektör temsilini yazdırma (şu an kapalı)

# TF-IDF değerlerini DataFrame'e çevirme (tablolu görünüm için)
df_tfidf = pd.DataFrame(vector_temsili, columns=feature_names)

# Her kelimenin ortalama TF-IDF değerini hesaplama (tüm belgeler boyunca)
tfidf=df_tfidf.mean(axis=0)

# Sonuçları ekrana yazdırma
print(tfidf)








