"""
Stemming ve Lemmatization - Doğal Dil İşleme Teknikleri
Bu dosya İngilizce kelimeler üzerinde kök bulma işlemlerini gösterir.
Stemming: Kelimenin sonundaki ekleri keserek kök bulur (hızlı ama bazen yanlış)
Lemmatization: Kelimeleri sözlüksel kök haline getirir (yavaş ama doğru)
"""

import nltk  # Doğal dil işleme için ana kütüphane
nltk.download('wordnet')  # Lemmatization için İngilizce sözlük veritabanını indirir
nltk.download('omw-1.4')  # WordNet için çok dilli destek paketini indirir

from nltk.stem import PorterStemmer  # Porter algoritması ile stemming yapan sınıf
from nltk.stem import WordNetLemmatizer  # WordNet sözlüğü ile lemmatization yapan sınıf

# STEMMING İŞLEMİ - Kural tabanlı kök bulma
stemmer = PorterStemmer()  # Porter stemming algoritması nesnesi oluştur
words = ["running", "runner", "ran", "runs", "better", "go", "went"]  # Test kelimeleri listesi

# List comprehension ile her kelimeye stemming uygula
stems = [stemmer.stem(w) for w in words]  # Her kelimeyi kökle ve yeni liste oluştur

# LEMMATIZATION İŞLEMİ - Sözlük tabanlı kök bulma
lemmatizer = WordNetLemmatizer()  # WordNet lemmatizer nesnesi oluştur
# pos='v' parametresi kelimelerin fiil olduğunu belirtir (daha doğru sonuç için)
lemmatized_words = [lemmatizer.lemmatize(w, pos='v') for w in words]  # Her kelimeyi lemmatize et

# SONUÇLARI EKRANA YAZDIRMA
print(f"Original words: {words}")  # Orijinal kelime listesini göster
print(f"Lemmatized words: {lemmatized_words}")  # Lemmatize edilmiş kelimeleri göster

"""
GENEL AÇIKLAMA - STEMMING VE LEMMATIZATION KARŞILAŞTIRMASI:

Bu kod doğal dil işlemede en temel ön işleme tekniklerinden olan kök bulma yöntemlerini gösterir.

STEMMING (Kök Bulma):
- Hızlı çalışır, basit kurallarla ekleri keser
- "running" -> "run", "better" -> "bett" (yanlış)
- Anlamsal doğruluğu garanti etmez

LEMMATIZATION (Gövdeleme):
- Yavaş çalışır, sözlük kullanır
- "running" -> "run", "better" -> "good" (doğru)
- Anlamsal olarak doğru sonuçlar verir

KULLANIM ALANLARI:
- Arama motorları: Stemming (hız öncelikli)
- Metin sınıflandırma: Lemmatization (doğruluk öncelikli)
- Büyük veri: Stemming (performans için)
"""
    