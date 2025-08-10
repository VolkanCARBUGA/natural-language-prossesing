# Kelime Anlam Belirsizliği Çözümü - Lesk Algoritması ile Word Sense Disambiguation

import nltk  # Doğal dil işleme kütüphanesi
from nltk.wsd import lesk  # Lesk algoritması ile kelime anlam belirsizliği çözümü için

# Gerekli NLTK veri setlerini indirme
nltk.download('wordnet')  # WordNet sözcük veri tabanı
nltk.download('omw-1.4')  # Açık çok dilli WordNet desteği
nltk.download('punkt')  # Cümle ve kelime tokenizasyonu için

# Test cümleleri - aynı kelime farklı anlamlarda kullanılıyor
sentence1="I go to the bank to deposit money"  # "bank" = finansal kurum anlamında
sentence2="The river bank is flooded after the heavy rain"  # "bank" = nehir kenarı anlamında

word1="bank"  # İlk cümlede analiz edilecek kelime
word2="bank"  # İkinci cümlede analiz edilecek kelime

# Lesk algoritması ile kelime anlamlarını belirleme
sense1=lesk(nltk.word_tokenize(sentence1),word1 )  # İlk cümledeki "bank" kelimesinin anlamını bulma
sense2=lesk(nltk.word_tokenize(sentence2),word2)  # İkinci cümledeki "bank" kelimesinin anlamını bulma

# Sonuçları yazdırma
print(f"Cümle1: {sentence1}")  # İlk cümleyi yazdırma
print(f"Kelime1: {word1}")  # Analiz edilen kelimeyi yazdırma
print(f"Anlam1: {sense1.definition()}")  # Bulunan anlamın tanımını yazdırma

print(f"Cümle2: {sentence2}")  # İkinci cümleyi yazdırma
print(f"Kelime2: {word2}")  # Analiz edilen kelimeyi yazdırma
print(f"Anlam2: {sense2.definition()}")  # Bulunan anlamın tanımını yazdırma

# KONU AÇIKLAMASI:
# Kelime Anlam Belirsizliği (Word Sense Disambiguation - WSD), bir kelimenin
# farklı bağlamlarda farklı anlamlar taşıyabileceği durumları çözmek için kullanılır.
# Lesk algoritması, bir kelimenin doğru anlamını bulmak için bağlamındaki
# diğer kelimelerle WordNet'teki tanımlar arasındaki örtüşmeyi hesaplar.
# Bu örnekte "bank" kelimesi iki farklı anlamda (finansal kurum ve nehir kenarı) kullanılmıştır.









