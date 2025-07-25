from  sklearn.feature_extraction.text import CountVectorizer  # Scikit-learn kütüphanesinden metin vektörleştirme aracını import ediyoruz


# örnek  bir metin
documents=[  # İki örnek Türkçe cümle içeren liste tanımlıyoruz
    "Bu çalışma  Ngram  çalışmasıdır",  # İlk örnek cümle - Ngram kavramını tanıtıyor
    "Bu çalışma  bir doğal dil işleme çalışmasıdır"  # İkinci örnek cümle - doğal dil işleme konusunu açıklıyor
]
vectorizer_unigram=CountVectorizer(ngram_range=(1,1))  # Unigram (1-gram) vektörleştirici oluşturuyoruz - tek kelime kombinasyonları
vectorizer_bigram=CountVectorizer(ngram_range=(2,2))   # Bigram (2-gram) vektörleştirici oluşturuyoruz - iki kelime kombinasyonları
vectorizer_trigram=CountVectorizer(ngram_range=(3,3))  # Trigram (3-gram) vektörleştirici oluşturuyoruz - üç kelime kombinasyonları

# unigram
X_unigram=vectorizer_unigram.fit_transform(documents)  # Unigram vektörleştiriciyi eğitip metinleri vektörlere dönüştürüyoruz
unigram_feature_names=vectorizer_unigram.get_feature_names_out()  # Unigram özellik isimlerini (kelimeler) alıyoruz

# bigram
X_bigram=vectorizer_bigram.fit_transform(documents)  # Bigram vektörleştiriciyi eğitip metinleri vektörlere dönüştürüyoruz
bigram_feature_names=vectorizer_bigram.get_feature_names_out()   # Bigram özellik isimlerini (kelime çiftleri) alıyoruz

# trigram
X_trigram=vectorizer_trigram.fit_transform(documents)  # Trigram vektörleştiriciyi eğitip metinleri vektörlere dönüştürüyoruz
trigram_feature_names=vectorizer_trigram.get_feature_names_out()  # Trigram özellik isimlerini (üçlü kelime grupları) alıyoruz

# sonuçların incelenmesi
print(f"Unigram: {unigram_feature_names}")  # Tek kelimeleri (unigram) yazdırıyoruz
print(f"Bigram: {bigram_feature_names}")    # Kelime çiftlerini (bigram) yazdırıyoruz
print(f"Trigram: {trigram_feature_names}")  # Üçlü kelime gruplarını (trigram) yazdırıyoruz

"""
N-GRAM KAVRAMI AÇIKLAMASI:

N-gram, doğal dil işlemede kullanılan temel bir tekniktir. Bir metindeki ardışık 
n adet kelimenin bir arada ele alınmasıdır.

1. UNIGRAM (1-gram): 
   - Tek tek kelimeleri analiz eder
   - En basit n-gram türüdür
   - Örnek: "Bu çalışma" → ["Bu", "çalışma"]

2. BIGRAM (2-gram):
   - İki ardışık kelimeyi birlikte analiz eder
   - Kelimeler arası basit ilişkileri yakalar
   - Örnek: "Bu çalışma Ngram" → ["Bu çalışma", "çalışma Ngram"]

3. TRIGRAM (3-gram):
   - Üç ardışık kelimeyi birlikte analiz eder
   - Daha karmaşık dil kalıplarını yakalar
   - Örnek: "Bu çalışma Ngram çalışmasıdır" → ["Bu çalışma Ngram", "çalışma Ngram çalışmasıdır"]

N-GRAM KULLANIM ALANLARI:
- Metin sınıflandırma
- Dil modelleme
- Makine çevirisi
- Metin madenciliği
- Spam tespiti
- Duygu analizi

Bu kod örneği, aynı metinlerin farklı n-gram seviyelerinde nasıl farklı 
özellikler ürettiğini göstermektedir.
"""






