"""
Tokenization - Metin Parçalama İşlemi
Bu dosya metinleri kelime ve cümle seviyesinde parçalama işlemini gösterir.
Tokenization: Sürekli metni anlamlı parçalara (token) ayırma işlemidir.
"""

import nltk  # Doğal dil işleme ana kütüphanesi
nltk.download('punkt')  # Punkt tokenizer modelini indir (kelime/cümle ayırma için)

text = "Hello, world! This is a test sentence."  # Tokenization için örnek İngilizce metin

# KELİME TOKENİZASYONU - Metni kelimelere ayırma
tokens = nltk.word_tokenize(text)  # Metni kelime seviyesinde parçalara ayır
# Sonuç: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', 'sentence', '.']
# Noktalama işaretleri de ayrı token olarak ele alınır

# CÜMLE TOKENİZASYONU - Metni cümlelere ayırma  
sentences = nltk.sent_tokenize(text)  # Metni cümle seviyesinde parçalara ayır
# Sonuç: ['Hello, world!', 'This is a test sentence.']
# Noktalama işaretlerine göre cümle sınırları belirlenir

# SONUÇLARI EKRANA YAZDIRMA
print(f"Original text: {text}")  # Orijinal metni göster
print(f"Tokenized sentences: {sentences}")  # Cümlelere ayrılmış halini göster

"""
GENEL AÇIKLAMA - TOKENİZASYON TEMELLERİ:

Bu kod doğal dil işlemenin en temel adımı olan tokenization işlemini gösterir.

TOKENİZASYON NEDİR?
- Sürekli metni anlamlı birimler (token) halinde ayırma
- Bilgisayarın metni anlayabilmesi için ilk gerekli adım
- Her token bir vektör elemanı olur

İKİ TEMEL TÜR:
1. KELİME TOKENİZASYONU:
   - Metni kelime seviyesinde ayırır
   - Noktalama işaretleri ayrı token olur
   - En yaygın kullanılan yöntem

2. CÜMLE TOKENİZASYONU:
   - Metni cümle seviyesinde ayırır
   - Noktalama kurallarını kullanır
   - Metin özetleme için kullanışlı

NLKT PUNKT TOKENIZER ÖZELLİKLERİ:
- 17 dil için eğitilmiş model
- Kısaltmaları tanır (Dr., Prof., vs.)
- Ondalık sayıları bölmez (3.14)
- Bağlamsal noktalama analizi yapar

KULLANIM ALANLARI:
- Metin sınıflandırma: Kelime tokenization
- Makine çevirisi: Alt-kelime tokenization  
- Duygu analizi: Kelime tokenization
- Özetleme: Cümle tokenization
"""