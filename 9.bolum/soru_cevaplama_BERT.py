"""
9. BÖLÜM - SORU CEVAPLAMA SİSTEMLERİ - BERT YAKLAŞIMI

Bu dosya BERT modeli ile extractive soru-cevaplama sistemi oluşturur.

SORU-CEVAPLAMA TÜRLERİ:
1. EXTRACTIVE: Verilen metinden cevap çıkarır (Bu projede kullanılan)
2. GENERATIVE: Yeni cevaplar üretir

BERT Q&A YAKLAŞIMI:
- Stanford Question Answering Dataset (SQuAD) ile eğitilmiş
- İki çıktı verir: start position, end position
- Context içinden cevabın başlangıç ve bitiş noktalarını bulur
- Çok yüksek doğruluk oranı (%90+)

ÇALIŞMA PRENSİBİ:
1. Soru ve context'i birleştir
2. BERT ile encode et
3. Start/end position skorları hesapla
4. En yüksek skorlu pozisyonları seç
5. Token'ları metne çevir

AVANTAJLARI:
✓ Çok yüksek doğruluk
✓ Hızlı inference
✓ Güvenilir cevaplar
✓ Çoklu dil desteği

DEZAVANTAJLARI:
✗ Sadece verilen metindeki bilgileri kullanır
✗ Context dışı bilgilere erişemez
✗ Yaratıcı cevaplar üretemez
✗ Uzun metinlerde performans düşer

KULLANIM ALANLARI:
- FAQ sistemleri
- Doküman arama
- Müşteri destek chatbotları
- Eğitim platformları
- Hukuki doküman analizi
"""

# BERT ile soru-cevaplama sistemi için gerekli kütüphaneler
from transformers import BertTokenizer, BertForQuestionAnswering  # BERT Q&A modeli
import torch  # PyTorch framework
import warnings  # Uyarı mesajları kontrolü

# Uyarı mesajlarını gizle (temiz çıktı için)
warnings.filterwarnings("ignore")

# Önceden eğitilmiş BERT Q&A modeli (SQuAD veri seti ile fine-tune edilmiş)
# SQuAD: Stanford Question Answering Dataset
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# BERT tokenizer'ını yükle (metinleri token'lara ayırmak için)
tokenizer = BertTokenizer.from_pretrained(model_name)

# BERT soru-cevaplama modelini yükle
model = BertForQuestionAnswering.from_pretrained(model_name)


# Soru-cevaplama fonksiyonu: soru ve context verilerek cevap üretir
def question_answer(question, context):
    # Soru ve context'i BERT formatına encode et
    encoding = tokenizer.encode_plus(
        question,           # Soruyu ekle
        context,           # Context'i ekle
        max_length=512,    # Maksimum token sayısı (BERT limiti)
        truncation=True,   # Uzun metinleri kes
        return_tensors="pt", # PyTorch tensor formatında döndür
    )
    # Token ID'lerini al
    input_ids=encoding["input_ids"]
    # Attention mask'ı al (hangi token'lara dikkat edilecek)
    attention_mask=encoding["attention_mask"]
    
    # Gradyan hesaplama olmadan (inference modu)
    with torch.no_grad():
        # Model ile tahmin yap: başlangıç ve bitiş skorları
        start_scores,end_scores=model(input_ids,attention_mask=attention_mask,return_dict=False)
        # En yüksek skora sahip başlangıç pozisyonu
        start_index=torch.argmax(start_scores,dim=1).item()
        # En yüksek skora sahip bitiş pozisyonu  
        end_index=torch.argmax(end_scores,dim=1).item()
        # Cevap token'larını al (başlangıçtan bitişe kadar)
        answer_tokens=tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index+1])
        # Token'ları tekrar metne çevir
        answer=tokenizer.convert_tokens_to_string(answer_tokens)
        return answer

# Örnek soru ve context
question="What is the capital of France?"  # Soru: Fransa'nın başkenti nedir?
context="France is a country in Europe. The capital of France is Paris."  # Context: Fransa hakkında bilgi

# Soru-cevaplama işlemini gerçekleştir
answer=question_answer(question,context)
# Cevabı yazdır
print(answer)



    
