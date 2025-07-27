# MAXIMUM ENTROPY MODEL - Maksimum Entropi ile Metin Sınıflandırma
# Bu dosya MaxEnt algoritmasını kullanarak duygu analizi örneğini gösterir

from nltk.classify import MaxentClassifier  # NLTK'dan Maximum Entropy sınıflandırıcı

# EĞİTİM VERİSİ HAZIRLAMA
# Her örnek (özellik_sözlüğü, sınıf_etiketi) formatında
train_data = [
    # POZİTİF DUYGU ÖRNEKLERİ
    ({"Love": True, "amazing": True, "happy": True, "terrible": False}, "positive"),  # Pozitif kelimeler içeren örnek
    ({"joy": True, "happy": True, "hate": False}, "positive"),  # Sevinç ve mutluluk ifade eden örnek
    
    # NEGATİF DUYGU ÖRNEKLERİ
    ({"hate": True, "terrible": True}, "negative"),  # Nefret ve korkunçluk ifade eden örnek
    ({"sad": True, "terrible": True, "love": False}, "negative"),  # Üzüntü ve olumsuzluk ifade eden örnek
]
# Özellikler boolean değerler şeklinde: True=kelime var, False=kelime yok

# MAXIMUM ENTROPY SINIFLANDIRICI EĞİTME
classifier = MaxentClassifier.train(train_data, max_iter=100)  # 100 iterasyon ile modeli eğit
# MaxEnt algoritması özelliklerin ağırlıklarını öğrenir

# TEST CÜMLESİ TANIMLAMA
test_sentence = "I do not like  this movie"  # Test edilecek olumsuz cümle

# TEST CÜMLESİ İÇİN ÖZELLİK ÇIKARIMI
# Test cümlesindeki kelimelerin varlığını kontrol etme
features = {
    word: (word in test_sentence.lower().split())  # Her kelime için True/False değeri
    for word in ["Love", "amazing", "happy", "terrible", "hate", "joy", "sad"]  # Bilinen özellik kelimeleri
}
# Çıktı örneği: {'Love': False, 'amazing': False, 'happy': False, 'terrible': False, 'hate': False, 'joy': False, 'sad': False}
# "like" kelimesi eğitim setinde olmadığı için algılanmıyor

# SINIFLANDIRMA İŞLEMİ
prediction = classifier.classify(features)  # Özelliklere dayalı sınıf tahmini

# SONUCU EKRANA YAZDIRMA
print(f"Test cümlesi: '{test_sentence}'")  # Test edilen cümle
print(f"Tahmin edilen duygu: {prediction}")  # Model tahmini: positive veya negative

"""
=== MAXIMUM ENTROPY MODEL (MAXENT) DETAYLI AÇIKLAMA ===

Maximum Entropy (MaxEnt) modeli, makine öğrenmesinde kullanılan güçlü bir 
sınıflandırma algoritmasıdır. Lojistik regresyonun genelleştirilmiş halidir.

MAXIMUM ENTROPY PRENSİBİ:

1. ENTROPI KAVRAMI:
   - Entropi = belirsizlik ölçüsü
   - Yüksek entropi = eşit dağılım (belirsiz)
   - Düşük entropi = eğik dağılım (kesin)

2. MAKSİMUM ENTROPİ İLKESİ:
   - Verilen kısıtlamalar altında entropiyi maksimize et
   - En az varsayım yap (önyargısızlık)
   - Sadece gözlemlenen veriye güven

ALGORITMA ÇALIŞMA PRENSİBİ:

1. ÖZELLİK FONKSİYONLARI:
   - Her özellik bir fonksiyon: f(x,y)
   - Örnek: f_love(x,y) = 1 if "love" in x and y="positive", else 0
   - Binary özellikler: kelime var/yok

2. AĞIRLIK ÖĞRENME:
   - Her özellik için ağırlık (λ) öğrenilir
   - Pozitif ağırlık = pozitif etkisi
   - Negatif ağırlık = negatif etkisi

3. OLASILIK HESAPLAMA:
   - P(y|x) = exp(Σ λᵢ fᵢ(x,y)) / Z(x)
   - Z(x) = normalizasyon sabiti
   - Softmax fonksiyonu benzer

ÖRNEK ÇALIŞMA:

Eğitim Verisi Analizi:
- "love" + "positive" → Güçlü pozitif ağırlık
- "hate" + "negative" → Güçlü negatif ağırlık
- "terrible" + "negative" → Güçlü negatif ağırlık

Test: "I do not like this movie"
- Bilinen pozitif kelime yok → düşük pozitif skor
- Bilinen negatif kelime yok → düşük negatif skor
- Model varsayılan sınıfa yönelir

AVANTAJLARI:

1. ÖZELLIK ESNEKLIGI:
   - Herhangi türde özellik kullanabilir
   - Boolean, sayısal, kategorik özellikler
   - Özellik kombinasyonları

2. İYİ GENELLEŞTİRME:
   - Overfitting'e dirençli
   - Az veri ile çalışabilir
   - Düzenlenmiş (regularized) öğrenme

3. YORUMLANABILIRLIK:
   - Ağırlıklar özellik önemini gösterir
   - Karar mekanizması anlaşılabilir

DEZAVANTAJLARI:

1. HESAPLAMA MALIYETI:
   - Iteratif optimizasyon gerekir
   - Büyük özellik setlerinde yavaş
   - Bellek kullanımı yüksek

2. ÖZELLİK MÜHENDİSLİĞİ:
   - İyi özellik seçimi kritik
   - Domain bilgisi gerekir
   - Manuel özellik çıkarımı

KARŞILAŞTIRMA:

NAIVE BAYES vs MAXENT:
- Naive Bayes: Özellik bağımsızlığı varsayar
- MaxEnt: Özellik etkileşimlerini modelleyebilir

SVM vs MAXENT:
- SVM: Margin maksimizasyonu
- MaxEnt: Entropi maksimizasyonu

KULLANIM ALANLARI:

1. DUYGU ANALİZİ:
   - Film/ürün yorumu sınıflandırma
   - Sosyal medya analizi

2. SPAM FİLTRELEME:
   - E-posta sınıflandırma
   - SMS spam tespiti

3. NAMED ENTITY RECOGNITION:
   - Kişi, yer, organizasyon tanıma
   - Biomedical entity recognition

4. POS TAGGING:
   - Kelime türü etiketleme
   - Syntax analizi

Bu örnekte model çok basit ama gerçek uygulamalarda binlerce özellik
ve milyonlarca örnek ile eğitilir.
"""
