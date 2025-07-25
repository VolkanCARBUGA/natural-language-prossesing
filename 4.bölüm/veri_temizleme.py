"""
Veri Temizleme - Metin Ön İşleme Teknikleri
Bu dosya ham metinleri makine öğrenmesi için hazırlama işlemlerini gösterir.
Amaç: Gürültülü verileri temizleyerek model performansını artırmak
"""

import string  # Python'un yerleşik string sabitleri ve fonksiyonları
import re  # Düzenli ifadeler (regex) için modül
from textblob import TextBlob  # Metin analizi ve yazım düzeltme kütüphanesi
from bs4 import BeautifulSoup  # HTML/XML ayrıştırma kütüphanesi

# 1. FAZLA BOŞLUKLARI TEMİZLEME
text = "Bu   metin   fazla   boşluklar   içeriyor.  Lütfen  temizleyin."  # Çoklu boşluk içeren test metni
cleaned_text = ' '.join(text.split())  # split() ardışık boşlukları böler, join() tek boşlukla birleştirir
# Çalışma mantığı: split() -> ['Bu', 'metin', 'fazla', ...], join(' ') -> "Bu metin fazla ..."

# 2. BÜYÜK HARFLERİ KÜÇÜK HARFE ÇEVİRME
text = "Bu MeTİN BÜyÜK HARfLeR İçERiYOR."  # Karışık büyük-küçük harf içeren test metni
cleaned_text = text.lower()  # Tüm karakterleri küçük harfe dönüştür
# Neden gerekli: "MERHABA" ve "merhaba" aynı kelime olarak algılanmalı

# 3. NOKTALAMA İŞARETLERİNİ KALDIRMA
text = "Bu metin, noktalama işaretleri! içeriyor?"  # Noktalama işaretli test metni
# string.punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ karakterlerini içerir
cleaned_text = text.translate(str.maketrans('', '', string.punctuation))  # Noktalama işaretlerini sil
# translate() fonksiyonu karakter değiştirme tablosu kullanır, burada silme için kullanılır

# 4. ÖZEL KARAKTERLERİ KALDIRMA
text = "Bu metin @#&* özel karakterler içeriyor!"  # Özel karakter içeren test metni
# Regex deseni: [^A-Za-z0-9\s] = harf, rakam, boşluk OLMAYAN her şey
cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Sadece harf, rakam, boşluk bırak
# re.sub(desen, değiştirme, metin) fonksiyonu desenle eşleşenleri değiştirir

# 5. YAZIM HATALARINI DÜZELTME (İngilizce)
text = "Ths is a smple sentnce with sme speling erors."  # Yazım hatası içeren İngilizce test metni
blob = TextBlob(text)  # TextBlob nesnesi oluştur (metin analizi için)
corrected_text = blob.correct()  # İstatistiksel yöntemlerle yazım hatalarını düzelt
# TextBlob İngilizce için eğitilmiş model kullanır, Türkçe için kısıtlıdır

# 6. HTML ETİKETLERİNİ TEMİZLEME
html = "<html><body><p>This is a <b>bold</b> paragraph.</p></body></html>"  # HTML etiketli test metni
soup = BeautifulSoup(html, 'html.parser')  # HTML'i ayrıştırmak için BeautifulSoup nesnesi
cleaned_html = soup.get_text()  # Sadece metin içeriğini al, HTML etiketlerini yok say
# get_text() tüm HTML etiketlerini kaldırıp sadece içerikteki metni döndürür

# SONUÇLARI GÖSTER
print(f"Original HTML: {html}")  # Orijinal HTML içeriği
print(f"Cleaned HTML: {cleaned_html}")  # Etiketsiz temiz metin

"""
GENEL AÇIKLAMA - METİN VERİ TEMİZLEME TEKNİKLERİ:

Bu kod makine öğrenmesi öncesi kritik olan veri temizleme sürecini gösterir.

VERİ TEMİZLEME NEDİR?
- Ham metinlerdeki gürültüyü kaldırma işlemi
- Model performansını %20-40 artırabilir
- Veri kalitesi = model kalitesi prensibi

TEMEL TEMİZLEME ADIMLARI:
1. Fazla boşluk temizleme: Tutarlılık için
2. Büyük harf normalleştirme: Aynı kelimeleri birleştirme
3. Noktalama kaldırma: Gürültü azaltma
4. Özel karakter temizleme: Standartlaştırma
5. Yazım düzeltme: Doğruluk artırma
6. HTML temizleme: Web verisi için

REGEX DESENLERI:
- [^A-Za-z0-9\s]: Harf, rakam, boşluk olmayan her şey
- \d+: Bir veya daha fazla rakam
- \s+: Bir veya daha fazla boşluk karakteri

DIKKAT EDİLECEKLER:
- Aşırı temizleme bilgi kaybına neden olur
- Domain-specific kurallar gerekebilir
- Dil özelliklerini göz önünde bulundur
- Temizleme sonrası veri kontrolü yap

KULLANIM SIRASI:
1. HTML temizleme (varsa)
2. Büyük harf normalleştirme
3. Özel karakter/noktalama temizleme
4. Fazla boşluk temizleme
5. Yazım düzeltme (son adım)
"""


