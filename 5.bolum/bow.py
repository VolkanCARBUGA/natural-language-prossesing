# Bag of Words (BOW) - Temel Kelime Torbalama Modeli
from sklearn.feature_extraction.text import CountVectorizer  # Metni sayısal vektörlere çeviren sınıf

# ÖRNEK VERİ SETİ OLUŞTURMA
documents = [
    "kedi bahçede",  # 1. belge: 2 kelime içeren kısa Türkçe cümle
    "kedi evde"      # 2. belge: 2 kelime içeren kısa Türkçe cümle
]
# Bu mini dataset 2 belge, 3 benzersiz kelime içerir: ["bahçede", "evde", "kedi"]

# VECTORIZER TANIMLAMASI
vectorizer = CountVectorizer()  # Varsayılan parametrelerle CountVectorizer nesnesi oluştur
# Varsayılan ayarlar: lowercase=True, stop_words=None, min_df=1, max_df=1.0

# METİNLERİ SAYISAL VEKTÖRLERE ÇEVİRME
X = vectorizer.fit_transform(documents)  # Belgelerden kelime sözlüğü oluştur ve vektörize et
# fit_transform() iki işlemi birleştirir:
# 1. fit(): Belgelerden kelime sözlüğü oluştur
# 2. transform(): Belgeleri bu sözlüğe göre vektörize et

# KELIME SÖZLÜĞÜNÜ ALMA
feature_names = vectorizer.get_feature_names_out()  # Oluşturulan kelime sözlüğünü al
# Çıktı: ['bahçede', 'evde', 'kedi'] - alfabetik sırayla sıralı

# SPARSE MATRİSİ DENSE MATRİSE ÇEVİRME
vector_temsili = X.toarray()  # Sparse matrix'i normal array'e çevir
# Sonuç matrisi: [[1, 0, 1], [0, 1, 1]]
# Satır = belge, sütun = kelime, değer = kelime frekansı

# SONUÇLARI GÖSTER
print(vector_temsili)  # BOW matrisini ekrana yazdır

"""
GENEL AÇIKLAMA - BAG OF WORDS (BOW) MODELİ:

Bu kod en temel metin vektörleştirme yöntemi olan BOW modelini gösterir.

BOW MODELİ NEDİR?
- Metinleri sayısal vektörlere çeviren basit yöntem
- Her belgeyi kelime frekanslarının vektörü olarak temsil eder
- Kelime sırası önemli değil, sadece varlık/yokluk önemli

ÇALIŞMA PRENSİBİ:
1. Tüm belgelerden benzersiz kelime sözlüğü oluştur
2. Her belgeyi bu sözlükteki kelime sayıları ile temsil et
3. Sonuç: belge_sayısı × kelime_sayısı boyutunda matris

ÖRNEK ÇIKTI AÇIKLAMASI:
Kelime sırası: ['bahçede', 'evde', 'kedi']
Belge 1 [1, 0, 1]: 'bahçede' 1 kez, 'evde' 0 kez, 'kedi' 1 kez
Belge 2 [0, 1, 1]: 'bahçede' 0 kez, 'evde' 1 kez, 'kedi' 1 kez

AVANTAJLARI:
- Basit ve anlaşılır
- Hızlı hesaplama
- Yorumlanabilir sonuçlar
- Küçük veri setleri için yeterli

DEZAVANTAJLARI:
- Kelime sırası kaybolur
- Anlamsal ilişkiler ihmal edilir
- Büyük sözlükler → yüksek boyut
- Nadir kelimeler gürültü oluşturur

KULLANIM ALANLARI:
- Metin sınıflandırma (basit)
- Spam filtreleme
- Duygu analizi (temel)
- Belge benzerlik hesaplama

GELİŞTİRME ÖNERİLERİ:
- TF-IDF ağırlıklandırma kullan
- N-gram modelleri dene
- Stop words filtrele
- Min/max frekans sınırları koy
"""





