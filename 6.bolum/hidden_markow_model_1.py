# HIDDEN MARKOV MODEL (HMM) - Gizli Markov Modeli ile POS Tagging
# Bu dosya HMM algoritmasını kullanarak kelime türü etiketleme (POS tagging) işlemini gösterir

import nltk  # Doğal dil işleme ana kütüphanesi
from nltk.tag import hmm  # NLTK'dan Hidden Markov Model sınıfı

# EĞİTİM VERİSİ HAZIRLAMA
# Her tuple (kelime, kelime_türü) şeklinde etiketlenmiş verilerden oluşur
train_data = [
    [("I", "PRP"), ("am", "VBP"), ("a", "DT"), ("student", "NN")],  # Cümle 1: "I am a student" - Zamir, Fiil, Belirteç, İsim
    [("you", "PRP"), ("are", "VBP"), ("a", "DT"), ("teacher", "NN")]  # Cümle 2: "you are a teacher" - Zamir, Fiil, Belirteç, İsim
]
# POS Etiketleri: PRP=Kişi zamiri, VBP=Fiil(çoğul), DT=Belirteç, NN=İsim(tekil)

# HMM TRAINER OLUŞTURMA
trainer=hmm.HiddenMarkovModelTrainer()  # HMM eğitici nesnesini oluştur

# MODELİ EĞİTME
hmm_model=trainer.train(train_data)  # Eğitim verisini kullanarak HMM modelini eğit
# Model, kelime geçiş olasılıklarını ve gözlem olasılıklarını öğrenir

# TEST CÜMLESİ HAZIRLAMA
test_sentence="He is a teacher".split()  # Test cümlesini kelimelere ayır
# Çıktı: ['He', 'is', 'a', 'teacher'] - Etiketlenmemiş kelime listesi

# ETKATLEME İŞLEMİ (POS TAGGING)
tagged_sentence=hmm_model.tag(test_sentence)  # HMM modelini kullanarak kelimeleri etiketle
# Model Viterbi algoritmasını kullanarak en olası etiket serisini bulur

# SONUCU EKRANA YAZDIRMA
print("Etiketlenmiş cümle:", tagged_sentence)  # Sonucu göster: [('He', 'PRP'), ('is', 'VBP'), ...]

"""
=== HIDDEN MARKOV MODEL (HMM) DETAYLI AÇIKLAMA ===

Hidden Markov Model (Gizli Markov Modeli), ardışık verilerde gizli durumları 
tahmin etmek için kullanılan istatistiksel bir modeldir.

HMM'IN TEMEL BİLEŞENLERİ:

1. DURUMLAR (STATES):
   - Gizli durumlar: POS etiketleri (PRP, VBP, DT, NN)
   - Gözlemlenebilir değil, tahmin edilmesi gereken
   - Bu örnekte: kelime türleri

2. GÖZLEMLER (OBSERVATIONS):
   - Görünen veriler: kelimeler ("I", "am", "a", "student")
   - Doğrudan gözlemlenebilir
   - Bu örnekte: cümledeki kelimeler

3. GEÇİŞ OLASILIĞI (TRANSITION PROBABILITY):
   - Bir durumdan diğerine geçme olasılığı
   - P(POS₂|POS₁): Bir kelime türünden sonra başka bir kelime türü gelme olasılığı
   - Örnek: P(NN|DT) = Belirteçten sonra isim gelme olasılığı

4. YAYILIM OLASILIĞI (EMISSION PROBABILITY):
   - Bir durumda belirli bir gözlem görme olasılığı
   - P(kelime|POS): Belirli bir kelime türünde belirli bir kelime görme olasılığı
   - Örnek: P("student"|NN) = İsim kategorisinde "student" kelimesi görme olasılığı

HMM ÇALIŞMA PRENSİBİ:

1. EĞİTİM AŞAMASI:
   - Etiketli verilerden geçiş ve yayılım olasılıklarını öğrenir
   - Maksimum likelihood estimation kullanır
   - Eğitim verisindeki kalıpları yakalar

2. ETİKETLEME AŞAMASI:
   - Viterbi algoritması kullanılır
   - En yüksek olasılıklı etiket serisini bulur
   - Dinamik programlama ile optimize edilir

VİTERBİ ALGORİTMASI:
- Tüm olası etiket kombinasyonlarını değerlendirir
- En yüksek olasılıklı yolu bulur
- O(T×N²) karmaşıklığında (T=kelime sayısı, N=durum sayısı)

ÖRNEK ÇALIŞMA:
Test: "He is a teacher"
1. "He" → PRP olasılığı yüksek (eğitim verisinden)
2. PRP→VBP geçişi yüksek + "is"→VBP yayılımı yüksek
3. VBP→DT geçişi yüksek + "a"→DT yayılımı yüksek  
4. DT→NN geçişi yüksek + "teacher"→NN yayılımı yüksek

AVANTAJLARI:
- Bağlamsal bilgiyi kullanır
- Sıralı veri için uygun
- Yorumlanabilir sonuçlar
- Belirsizlikle başa çıkabilir

DEZAVANTAJLARI:
- Markov varsayımı (sadece önceki durum önemli)
- Eğitim verisi gerektirir
- Uzun mesafe bağımlılıkları yakalayamaz

KULLANIM ALANLARI:
- POS Tagging (kelime türü etiketleme)
- Named Entity Recognition (NER)
- Konuşma tanıma
- Biyoinformatik (DNA dizisi analizi)
- Zaman serisi analizi

Bu basit örnek HMM'in temel mantığını gösterir. Gerçek uygulamalarda
çok daha büyük eğitim verileri ve daha karmaşık modeller kullanılır.
"""