"""
9. BÖLÜM - MAKİNE ÇEVİRİSİ (MACHINE TRANSLATION)

Bu dosya Marian modeli kullanarak neural machine translation yapar.

MAKİNE ÇEVİRİSİ NEDİR?
- Metinleri bir dilden diğerine otomatik çevirme
- Küresel iletişim için kritik teknoloji
- İnsan kalitesine yakın çeviri hedefi

NEURAL MACHINE TRANSLATION (NMT):
- Derin öğrenme tabanlı çeviri yaklaşımı
- Sequence-to-sequence modeller kullanır
- Attention mekanizması ile uzun cümleler
- Context'i koruyarak çeviri yapar

MARIAN FRAMEWORK:
- C++ tabanlı, hızlı NMT toolkit
- Hugging Face ile entegre
- Çoklu dil çifti desteği
- OPUS veri seti ile eğitilmiş modeller

HELSINKI-NLP MODELLERİ:
- Açık kaynak çeviri modelleri
- 1000+ dil çifti
- OPUS-MT serisi
- Sürekli güncellenen modeller

ÇALIŞMA PRENSİBİ:
1. Kaynak metni tokenize et
2. Encoder ile source representation
3. Decoder ile target generation
4. Attention ile alignment
5. Beam search ile en iyi çeviri

AVANTAJLARI:
✓ Yüksek çeviri kalitesi
✓ Context'i korur
✓ İdiomatic ifadeler
✓ Çok dil desteği

DEZAVANTAJLARI:
✗ Domain spesifik terimler
✗ Kültürel referanslar
✗ Uzun metinlerde tutarsızlık
✗ Model boyutu büyük

KULLANIM ALANLARI:
- Çok dilli web siteleri
- Doküman çevirisi
- Anlık mesajlaşma
- E-ticaret platformları
- Haber çevirisi
"""

# Marian tabanlı çeviri sistemi için gerekli kütüphaneler
from  transformers import MarianMTModel, MarianTokenizer  # Marian çeviri modeli

# Helsinki-NLP'nin İngilizce->Almanca çeviri modeli
# opus-mt: OPUS veri seti ile eğitilmiş makine çevirisi modeli
model_name="Helsinki-NLP/opus-mt-en-de"
# Marian tokenizer'ını yükle (çeviri için özel tokenizer)
tokenizer=MarianTokenizer.from_pretrained(model_name)
# Marian çeviri modelini yükle
model=MarianMTModel.from_pretrained(model_name)

# Çevrilecek İngilizce metin
text="What is the capital of France?"

# Metni tokenize et ve modelle çevir
# return_tensors="pt": PyTorch tensor formatı, padding=True: eşit uzunluk için doldurma
translated_text=model.generate(**tokenizer(text,return_tensors="pt",padding=True))

# Çevrilmiş token'ları tekrar metne çevir ve özel token'ları temizle
print(tokenizer.decode(translated_text[0],skip_special_tokens=True))
# Ham çeviri token'larını da göster (debug için)
print("translated_text",translated_text)






