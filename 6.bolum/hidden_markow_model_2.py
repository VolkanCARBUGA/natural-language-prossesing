# HIDDEN MARKOV MODEL (HMM) - Gerçek Veri Seti ile Gelişmiş POS Tagging
# Bu dosya CONLL2000 veri seti kullanarak daha gerçekçi HMM uygulamasını gösterir

import nltk  # Doğal dil işleme ana kütüphanesi
from nltk.tag import hmm  # NLTK'dan Hidden Markov Model sınıfı
from nltk.corpus import conll2000  # CONLL2000 corpus'u - standart NLP veri seti

# GEREKLI VERI SETINI INDIRME
nltk.download("conll2000")  # CONLL2000 chunking veri setini indir
# Bu veri seti, Wall Street Journal'dan alınmış etiketlenmiş cümleler içerir

# EĞİTİM VE TEST VERİSİNİ YÜKLEME
train_data=conll2000.tagged_sents("train.txt")  # Eğitim verisi: ~8,900 etiketli cümle
test_data=conll2000.tagged_sents("test.txt")    # Test verisi: ~2,000 etiketli cümle
# Her cümle [(kelime, POS_etiketi), ...] formatında

# EĞİTİM VERİSİNDEN ÖRNEK GÖSTERME
print(f"Eğitim verisi örneği: {train_data[:1]}")  # İlk cümleyi göster
# Çıktı örneği: [[('Confidence', 'NN'), ('in', 'IN'), ('the', 'DT'), ('pound', 'NN'), ...]]

# HMM TRAINER OLUŞTURMA VE MODELİ EĞİTME
trainer=hmm.HiddenMarkovModelTrainer()  # HMM eğitici nesnesini oluştur
hmm_model=trainer.train(train_data)     # Büyük eğitim verisi ile modeli eğit
# Model 8,900 cümleden POS geçiş ve yayılım olasılıklarını öğrenir

# TEST CÜMLESİ HAZIRLAMA
test_sentence="I am a student".split()  # Basit test cümlesi
# Çıktı: ['I', 'am', 'a', 'student'] - Etiketlenmemiş kelime listesi

# ETIKETLEME İŞLEMİ GERÇEKLEŞTIRME
tagged_sentence=hmm_model.tag(test_sentence)  # HMM ile cümleyi etiketle
# Viterbi algoritması ile en olası POS etiket serisini bulur

# SONUÇLARI EKRANA YAZDIRMA
print("Test cümlesi: ",test_sentence)      # Orijinal kelimeler
print("Etiketlenmiş cümle: ",tagged_sentence)  # Etiketlenmiş sonuç

"""
=== GERÇEK VERİ SETİ İLE HMM - CONLL2000 VERİ SETİ AÇIKLAMASI ===

Bu kod, gerçek dünya NLP uygulamalarında kullanılan büyük ölçekli veri seti 
ile HMM modelinin nasıl eğitildiğini gösterir.

CONLL2000 VERİ SETİ ÖZELLİKLERİ:

1. VERİ KAYNAĞI:
   - Wall Street Journal korpusundan alınmış
   - Finansal haberler ve makaleler
   - Profesyonel editörler tarafından etiketlenmiş
   - NLP araştırmalarında standart benchmark

2. VERİ BOYUTU:
   - Eğitim: ~8,900 cümle, ~200,000 kelime
   - Test: ~2,000 cümle, ~45,000 kelime
   - 45+ farklı POS etiketi
   - Gerçek dil kullanım kalıplarını yansıtır

3. POS ETİKET ÖRNEKLERİ:
   - NN: Noun (isim) - "student", "book"
   - VBZ: Verb 3rd person singular - "runs", "goes"  
   - DT: Determiner (belirteç) - "the", "a"
   - JJ: Adjective (sıfat) - "good", "big"
   - IN: Preposition (edat) - "in", "on", "at"

BÜYÜK VERİ SETİ AVANTAJLARI:

1. DAHA DOĞRU OLASILIKLAR:
   - Daha fazla örnek → daha güvenilir istatistikler
   - Nadir kelime-etiket kombinasyonlarını öğrenir
   - Çeşitli cümle yapılarını kapsar

2. GELİŞMİŞ GENELLEŞTİRME:
   - Test verisi üzerinde daha iyi performans
   - Bilinmeyen kelimeleri daha iyi tahmin eder
   - Gerçek dünya metinlerinde daha başarılı

3. KARMAŞIK DİL KALIPLARI:
   - Uzun cümleler ve karmaşık yapılar
   - Çeşitli konu ve jargon
   - Farklı yazım stilleri

MODEL EĞİTİMİ SÜRECİ:

1. GEÇİŞ OLASILIĞI HESAPLAMA:
   - P(POS₂|POS₁) = Count(POS₁,POS₂) / Count(POS₁)
   - Örnek: P(NN|DT) = Belirteçten sonra isim gelme sıklığı

2. YAYILIM OLASILIĞI HESAPLAMA:
   - P(kelime|POS) = Count(kelime,POS) / Count(POS)
   - Örnek: P("student"|NN) = "student" kelimesinin isim olma sıklığı

3. SMOOTHING TEKNİKLERİ:
   - Görülmemiş kombinasyonlar için küçük olasılık atar
   - Sıfır olasılık problemini çözer

PERFORMANS BEKLENTİLERİ:

- İngilizce POS tagging için %95-97 doğruluk
- Bilinen kelimeler: %98+ doğruluk
- Bilinmeyen kelimeler: %85-90 doğruluk
- Karmaşık cümleler: %90-95 doğruluk

GERÇEK DÜNYA UYGULAMALARI:

1. ARAMA MOTORLARI:
   - "run" fiil mi isim mi? Bağlama göre anlama
   - Daha akıllı arama sonuçları

2. MAKİNE ÇEVİRİSİ:
   - Kelime türü hedef dilde doğru çeviriye yardımcı
   - Cümle yapısı korunur

3. BİLGİ ÇIKARIMI:
   - İsimler varlık olarak tanımlanır
   - Fiiller eylem olarak kategorize edilir

4. DİL ÖĞRETİMİ:
   - Öğrencilere kelime türlerini öğretme
   - Otomatik gramer kontrolü

Bu örnekte model, büyük veri seti sayesinde gerçek dil kullanımını
öğrenir ve test cümlesinde yüksek doğrulukla POS tagging yapar.
"""


