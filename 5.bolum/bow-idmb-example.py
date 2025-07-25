# IMDB Film Yorumları BOW Analizi - Gerçek Veri Seti Uygulaması

# GEREKLI KÜTÜPHANELERİ İÇE AKTARMA
import pandas as pd  # Veri manipülasyonu ve analizi için
import numpy as np  # Sayısal hesaplamalar için
import matplotlib.pyplot as plt  # Grafik çizimi için
import seaborn as sns  # İstatistiksel görselleştirme için
from sklearn.feature_extraction.text import CountVectorizer  # BOW vektörleştirme için
import re  # Düzenli ifadeler (regex) için
from collections import Counter  # Kelime sayma işlemleri için

# VERİ SETİNİ YÜKLEME
dataframe = pd.read_csv("5.bolum/IMDB Dataset.csv")  # IMDB film yorumları CSV dosyasını oku
# Dataset: 50,000 film yorumu (25,000 pozitif, 25,000 negatif)

# VERİ SETİNDEN GEREKLI SÜTUNLARI ALMA
documents = dataframe["review"]  # Metin yorumları sütunu (50,000 yorum)
labels = dataframe["sentiment"]  # Duygu etiketleri: 'positive' veya 'negative'

# METİN TEMİZLEME FONKSİYONU
def clean_text(text):
    """
    Ham metinleri BOW analizi için temizleyen fonksiyon
    """
    text = text.lower()  # Tüm harfleri küçük harfe çevir (normalizasyon)
    
    text = re.sub(r"\d+", " ", text)  # Tüm rakamları boşlukla değiştir
    # Regex r"\d+" = bir veya daha fazla ardışık rakam
    
    text = re.sub(r"[^\w\s]", " ", text)  # Harf ve boşluk dışındaki karakterleri sil
    # Regex r"[^\w\s]" = kelime karakteri (\w) veya boşluk (\s) OLMAYAN her şey
    
    # 3 karakterden kısa kelimeleri filtrele (gürültü azaltma)
    text = " ".join([word for word in text.split() if len(word) > 3])
    
    return text  # Temizlenmiş metni döndür

# TÜM BELGELERİ TEMİZLEME
cleaned_documents = [clean_text(doc) for doc in documents]  # Her yorumu temizle
# List comprehension ile 50,000 yorumun hepsini temizle

# BAG OF WORDS MODELİ OLUŞTURMA
vectorizer = CountVectorizer()  # Varsayılan parametrelerle BOW vektörleştirici

# SADECE İLK 75 BELGEYİ VEKTÖRLEŞTİRME (performans için)
X = vectorizer.fit_transform(cleaned_documents[:75])  # İlk 75 yorumu sayısal vektörlere çevir
# fit_transform() = kelime sözlüğü oluştur + vektörize et

# KELİME SÖZLÜĞÜNÜ ALMA
feature_names = vectorizer.get_feature_names_out()  # Oluşturulan kelime sözlüğünü al
# Bu sözlük 75 belgeden çıkarılan benzersiz kelimeleri içerir

# SPARSE MATRİSİ DENSE MATRİSE ÇEVİRME
vector_temsili_2 = X.toarray()  # Sparse matrix'i normal numpy array'e çevir
# Boyut: (75, kelime_sayısı) - her satır bir belge, her sütun bir kelime

# PANDAS DATAFRAME OLUŞTURMA (görselleştirme için)
df_bow = pd.DataFrame(vector_temsili_2, columns=feature_names)  # BOW matrisini DataFrame'e çevir

# KELİME FREKANS ANALİZİ
word_counts = X.sum(axis=0)  # Her kelimenin toplam geçiş sayısını hesapla
# axis=0: sütunlar boyunca toplama (tüm belgelerde her kelimenin toplamı)

word_freq = dict(zip(feature_names, word_counts))  # Kelime-frekans sözlüğü oluştur
# zip() kelime isimlerini frekanslarla eşleştirir

# EN ÇOK GEÇEN KELİMELERİ BULMA
most_common_words = Counter(word_freq).most_common(5)  # En sık geçen 5 kelimeyi bul
# Counter.most_common(n) en sık geçen n elemanı döndürür

# SONUÇLARI GÖSTER
print("En çok geçen 5 kelime: ", most_common_words)  # Sonuçları ekrana yazdır

"""
GENEL AÇIKLAMA - GERÇEK VERİ SETİ İLE BOW ANALİZİ:

Bu kod IMDB film yorumları veri seti ile gerçek dünya BOW uygulamasını gösterir.

VERİ SETİ ÖZELLİKLERİ:
- 50,000 film yorumu (çok büyük veri)
- İki sınıf: pozitif/negatif duygu
- Değişken uzunlukta metinler
- Gerçek kullanıcı yorumları (gürültülü)

VERİ TEMİZLEME ADIMLARI:
1. Küçük harfe çevirme: Normalizasyon
2. Rakam kaldırma: Sayılar genellikle anlamsız
3. Özel karakter temizleme: Gürültü azaltma
4. Kısa kelime filtresi: "a", "an", "is" gibi anlamsız kelimeler

PERFORMANS OPTİMİZASYONU:
- Sadece ilk 75 belge kullanıldı
- Gerçek uygulamada tüm veri kullanılır
- Batch processing gerekebilir

FREKANS ANALİZİ:
- En sık kelimeler genellikle stop words
- Orta frekanslı kelimeler daha anlamlı
- Nadir kelimeler gürültü olabilir

BEKLENEN SONUÇLAR:
- "movie", "film", "good", "great" gibi kelimeler sık geçer
- Pozitif/negatif kelimeler farklı frekanslarda olur
- BOW matrisi çok seyrek (sparse) olur

GELİŞTİRME ÖNERİLERİ:
1. TF-IDF kullan (kelime önemini ağırlıklandır)
2. Stop words filtrele
3. Min/max frekans sınırları koy
4. N-gram modeli dene
5. Daha fazla veri kullan

GERÇEK UYGULAMALAR:
- Spam filtreleme
- Duygu analizi
- Konu modelleme
- Belge sınıflandırma
- Metin benzerlik hesaplama
"""





