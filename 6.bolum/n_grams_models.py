# N-gram Dil Modelleri - Trigram Olasılık Hesaplama Örneği

import nltk  # Doğal dil işleme kütüphanesi
from nltk.util import ngrams  # N-gram oluşturmak için fonksiyon
from nltk.tokenize import word_tokenize  # Cümleleri kelimelere ayırmak için
from collections import Counter  # Frekans sayma için

# Eğitim verisi - 7 basit cümleden oluşan corpus
corpus = [
    "I love apple",      # Cümle 1
    "I love banana",     # Cümle 2
    "I love NLP",        # Cümle 3
    "you love me",       # Cümle 4
    "He loves apple",    # Cümle 5
    "They love NLP",     # Cümle 6
    "I love you and me", # Cümle 7
]

# Her cümleyi küçük harfe çevirip kelimelerine ayırma (tokenization)
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

# Bigram (2-gram) oluşturma - ardışık 2 kelimeli gruplar
bigrams = []
for token_list in tokens:  # Her cümlenin token listesi için
    bigrams.extend(list(ngrams(token_list, 2)))  # 2-gram'ları listeye ekle

# Bigram'ların frekanslarını sayma
bigrams_freq = Counter(bigrams)

# Trigram (3-gram) oluşturma - ardışık 3 kelimeli gruplar
trigrams = []
for token_list in tokens:  # Her cümlenin token listesi için
    trigrams.extend(list(ngrams(token_list, 3)))  # 3-gram'ları listeye ekle

# Trigram'ların frekanslarını sayma
trigrams_freq = Counter(trigrams)

# Koşullu olasılık hesaplama için temel bigram
bigram = ("i", "love")  # "I love" bigram'ı

# P(kelime|I,love) koşullu olasılıklarını hesaplama
# P(you|I,love) = Count(I,love,you) / Count(I,love)
prob_you = trigrams_freq[("i", "love", "you")] / bigrams_freq[bigram]

# P(apple|I,love) = Count(I,love,apple) / Count(I,love)
prob_apple = trigrams_freq[("i", "love", "apple")] / bigrams_freq[bigram]

# P(banana|I,love) = Count(I,love,banana) / Count(I,love)
prob_banana = trigrams_freq[("i", "love", "banana")] / bigrams_freq[bigram]

# P(nlp|I,love) = Count(I,love,nlp) / Count(I,love)
prob_nlp = trigrams_freq[("i", "love", "nlp")] / bigrams_freq[bigram]

# Hesaplanan olasılıkları ekrana yazdırma
print(f"P(you|I,love)={prob_you}")      # "I love" dan sonra "you" gelme olasılığı
print(f"P(apple|I,love)={prob_apple}")  # "I love" dan sonra "apple" gelme olasılığı
print(f"P(banana|I,love)={prob_banana}")# "I love" dan sonra "banana" gelme olasılığı
print(f"P(nlp|I,love)={prob_nlp}")      # "I love" dan sonra "nlp" gelme olasılığı

"""
N-GRAM DİL MODELLERİ HAKKINDA:

Bu kod, N-gram dil modellerinin temelini gösteren bir örnektir. 

N-gram modelleri:
- Dil modellemede kullanılan istatistiksel yöntemlerdir
- Bir kelimenin, kendinden önceki n-1 kelimeye bağlı olarak gelme olasılığını hesaplar
- Bu örnekte trigram (3-gram) modeli kullanılmıştır

Çalışma Mantığı:
1. Bigram'lar: Ardışık 2 kelimeli gruplar (örn: "i love", "love apple")
2. Trigram'lar: Ardışık 3 kelimeli gruplar (örn: "i love apple", "love you and")
3. Koşullu Olasılık: P(w3|w1,w2) = Count(w1,w2,w3) / Count(w1,w2)

Bu örnekte "I love" bigram'ından sonra hangi kelimenin gelme olasılığının 
daha yüksek olduğu hesaplanmıştır. Bu tür modeller:
- Metin tamamlama
- Makine çevirisi
- Konuşma tanıma
- Otomatik metin üretimi
gibi uygulamalarda kullanılır.

Sonuçlar corpus'taki frekansları yansıtır ve daha büyük veri setleriyle
daha güvenilir olasılıklar elde edilir.
"""






