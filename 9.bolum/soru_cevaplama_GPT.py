"""
9. BÖLÜM - SORU CEVAPLAMA SİSTEMLERİ - GPT-2 YAKLAŞIMI

Bu dosya GPT-2 modeli ile generative soru-cevaplama sistemi oluşturur.

GENERATIVE SORU-CEVAPLAMA:
- Verilen context'e dayanarak yeni cevaplar üretir
- Dil modeli yaklaşımı kullanır
- Yaratıcı ve açıklayıcı cevaplar verebilir
- BERT'ten farklı olarak sadece extraction yapmaz

GPT-2 YAKLAŞIMI:
- Generative Pre-trained Transformer
- Autoregressive dil modeli
- Next token prediction ile eğitilmiş
- Text completion görevi olarak soru-cevaplama

ÇALIŞMA PRENSİBİ:
1. Prompt formatı oluştur (Question + Context + Please answer)
2. GPT-2 ile text generation yap
3. Temperature ile yaratıcılık kontrolü
4. Generated text'ten cevap kısmını çıkar

BERT İLE KARŞILAŞTIRMA:
BERT (Extractive):
✓ Yüksek doğruluk
✓ Güvenilir cevaplar
✗ Sadece context'teki bilgiler

GPT-2 (Generative):
✓ Yaratıcı cevaplar
✓ Açıklayıcı yanıtlar
✗ Bazen yanlış bilgi üretir
✗ Daha az güvenilir

AVANTAJLARI:
✓ Esnek cevap formatları
✓ Detaylı açıklamalar
✓ Çoklu dil desteği
✓ Context'i genişletebilir

DEZAVANTAJLARI:
✗ Hallucination riski
✗ Tutarsız cevaplar
✗ Fact-checking gerektirir
✗ Yavaş inference

KULLANIM ALANLARI:
- Eğitim chatbotları
- Yaratıcı yazma asistanları
- Açıklayıcı FAQ sistemleri
- İnteraktif öğrenme platformları
"""

# GPT-2 ile soru-cevaplama sistemi için gerekli kütüphaneler
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # GPT-2 modeli ve tokenizer
import torch  # PyTorch framework

# GPT-2 temel modelini kullan (dil modelleme için eğitilmiş)
model_name="gpt2"

# GPT-2 tokenizer'ını yükle (metinleri token'lara ayırmak için)
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
# GPT-2 dil modelini yükle (metin üretimi için)
model=GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 ile soru-cevaplama fonksiyonu (generative yaklaşım)
def generate_answer(question,context):
    # Prompt formatı oluştur (soru ve context'i birleştir)
    input_text=f"Question: {question}\nContext: {context}\Please answer the question according to the context:"
    # Metni token'lara çevir
    inputs=tokenizer.encode(input_text,return_tensors="pt")
    
    # Gradyan hesaplama olmadan (inference modu)
    with torch.no_grad():
        # GPT-2 ile metin üret (sıcaklık=0.7 ile yaratıcılık)
        output=model.generate(inputs,max_length=512,do_sample=True,temperature=0.7)
        # Token'ları tekrar metne çevir
        answer=tokenizer.decode(output[0],skip_special_tokens=True)
        # "Answer:" kısmından sonrasını al (eğer varsa)
        answer=answer.split("Answer:")[-1].strip()
        return answer
    
# Örnek soru ve context
question="What is the capital of France?"  # Soru: Fransa'nın başkenti nedir?
context="France is a country in Europe. The capital of France is Paris."  # Context

# GPT-2 ile cevap üret
answer=generate_answer(question,context)
# Cevabı yazdır
print(answer)
