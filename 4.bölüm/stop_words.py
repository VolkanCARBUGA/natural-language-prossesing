"""
Stop Words - Gereksiz Kelimelerin Temizlenmesi
Bu dosya metinlerden anlam taşımayan sık kullanılan kelimeleri kaldırmayı gösterir.
Stop words: "ve", "ile", "bu", "the", "is" gibi cümlede anlam katmayan kelimeler
"""

import nltk  # Doğal dil işleme kütüphanesi
from nltk.corpus import stopwords  # Hazır stop words listeleri modülü
nltk.download('stopwords')  # 16 farklı dilde stop words veri setini indir

# İNGİLİZCE STOP WORDS İLE FİLTRELEME
stop_words = set(stopwords.words('english'))  # İngilizce stop words'ü küme olarak al (hızlı arama için)
text = "This is a sample sentence demonstrating the removal of stop words."  # Test metni
text_list = text.split()  # Metni boşluklardan bölerek kelime listesi oluştur

# List comprehension ile filtreleme: kelime küçük harfe çevrilip stop words'te yoksa al
filtered_text = [word for word in text_list if word.lower() not in stop_words]

# TÜRKÇE STOP WORDS İLE FİLTRELEME
stop_words = set(stopwords.words('turkish'))  # Türkçe stop words kümesi
text = "merhaba arkadaşlar çok güzel bir ders çalışması yapıyoruz."  # Türkçe test metni
text_list = text.split()  # Kelime listesine çevir

# Stop words'te olmayan kelimeleri filtrele
filtered_text = [word for word in text_list if word.lower() not in stop_words]

# MANUEL TÜRKÇE STOP WORDS LİSTESİ
# Kütüphane kullanmadan elle tanımlanmış Türkçe stop words
turkish_stop_words = [
    "ben", "sen", "o", "biz", "siz", "onlar", "bu", "şu", "ne", "nasıl",  # Zamirler
    "ki", "de", "da", "ile", "ve", "ama", "fakat", "çünkü", "ya",        # Bağlaçlar
    "ya da", "herhangi bir", "hiçbir"                                     # Belirteçler
]

text = "bu bir  örnek cümledir  ve biz bu cümlede  şu kelimeleri  kaldıracağız."  # Test metni
text_list = text.split()  # Kelime listesine böl

# Manuel liste ile filtreleme
filtered_text = [word for word in text_list if word.lower() not in turkish_stop_words]

# SONUÇLARI GÖSTER
print("Original text:", text)  # Orijinal metni ekrana yazdır
print("Filtered text:", ' '.join(filtered_text))  # Filtrelenmiş metni birleştirip yazdır

"""
GENEL AÇIKLAMA - STOP WORDS TEMİZLEME TEKNİKLERİ:

Bu kod metin ön işlemede kritik bir adım olan stop words temizleme işlemini gösterir.

STOP WORDS NEDİR?
- Cümlede anlam katmayan, sık kullanılan kelimeler
- Makine öğrenmesi modellerinde gürültü oluşturur
- Metin boyutunu azaltarak performansı artırır

ÜÇ FARKLI YAKLAŞIM:
1. NLTK İngilizce: 179 kelimelik hazır liste
2. NLTK Türkçe: 233 kelimelik hazır liste  
3. Manuel Liste: Projeye özel özelleştirilebilir

KULLANIM DURUMLARI:
- Metin sınıflandırma: Mutlaka kullan
- Duygu analizi: Dikkatli kullan (bazı stop words duygu taşır)
- Arama motorları: Kullanma (kullanıcı "the" arayabilir)
- Özetleme: Kullan (alakasız kelimeleri temizler)

PERFORMANS ETKİSİ:
- Metin boyutunu %30-50 azaltır
- İşlem hızını 2-3 kat artırır
- Model doğruluğunu genellikle artırır
"""
