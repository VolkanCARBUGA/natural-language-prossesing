"""
9. BÖLÜM - BİLGİ ERİŞİMİ (INFORMATION RETRIEVAL) - BERT TABANLI

Bu dosya BERT modeli kullanarak semantik bilgi erişimi sistemi oluşturur.

BİLGİ ERİŞİMİ NEDİR?
- Büyük doküman koleksiyonlarından ilgili bilgileri bulma süreci
- Arama motorları, doküman yönetim sistemleri için temel
- Kullanıcı sorgusuna en uygun dokümanları döndürür

BERT YAKLAŞIMI:
- Bidirectional Encoder Representations from Transformers
- Derin öğrenme tabanlı dil modeli
- Context-aware embeddings üretir
- Semantik benzerlik hesaplaması

ÇALIŞMA PRENSİBİ:
1. Sorgu ve dokümanları BERT ile vektörleştir
2. Cosine similarity ile benzerlik hesapla
3. En yüksek benzerlik skoruna sahip dokümanları döndür

AVANTAJLARI:
✓ Semantik anlam anlayışı
✓ Context-aware arama
✓ Çok dilli destek
✓ Pre-trained model kullanımı

DEZAVANTAJLARI:
✗ Yüksek hesaplama maliyeti
✗ Büyük model boyutu
✗ GPU gereksinimleri
✗ Uzun işlem süreleri

KULLANIM ALANLARI:
- Akademik makale arama
- Hukuki doküman analizi
- E-ticaret ürün arama
- Müşteri destek sistemleri
"""

# BERT ile bilgi erişimi (Information Retrieval) için gerekli kütüphaneler
from transformers import BertTokenizer, BertModel  # BERT modeli ve tokenizer
import numpy as np                                 # Sayısal işlemler için
from sklearn.metrics.pairwise import cosine_similarity  # Benzerlik hesaplama


# BERT modelinin adı - İngilizce temel model
model_name = "bert-base-uncased"

# BERT tokenizer'ını yükle (metinleri token'lara ayırmak için)
tokenizer = BertTokenizer.from_pretrained(model_name)
# BERT modelini yükle (embedding üretmek için)
model = BertModel.from_pretrained(model_name)

# Arama yapılacak doküman koleksiyonu (veritabanı)
documents = [
    "Machine learning is a field of artificial intelligence",
    "Natural language processing involves understanding human language", 
    "Artificial intelligence encomppases machine learning and natural language processing (nlp)",
    "Deep learning is a subset of machine learning",
    "Data science combines statistics, adta analysis and machine learning",
    "I go to shop",  # İlgisiz doküman (test için)
]
# Kullanıcının sorgusu
query="What is machine learning?"

# Metin için BERT embedding'i üreten fonksiyon
def get_embeddings(text):
    # Metni BERT formatına çevir (tokenize, padding, truncation)
    inputs=tokenizer(text,return_tensors="pt",padding=True,truncation=True)
    
    # BERT modeli ile forward pass yap
    outputs=model(**inputs)
    # Son gizli katman çıktısını al
    last_hidden_state=outputs.last_hidden_state
    # Ortalama embedding hesapla (tüm token'ların ortalaması)
    embedding=last_hidden_state.mean(dim=1)
    # PyTorch tensor'ını numpy array'e çevir
    return embedding.detach().numpy()

# Her doküman için embedding üret (list comprehension kullanarak)
doc_embeddings=[get_embeddings(doc) for doc in documents]
# Sorgu için embedding üret
query_embedding=get_embeddings(query)

# Doküman embedding'lerini numpy array'e çevir ve dikey olarak birleştir
doc_embeddings = np.vstack(doc_embeddings)

# Cosine similarity ile sorgu ve her doküman arasındaki benzerliği hesapla
similarities=cosine_similarity(query_embedding,doc_embeddings)
# Sonuçları görüntüle (benzerlik skorları ile)
for i,score in enumerate(similarities[0]):
    print(f"Document {i+1}: {documents[i]}")
    print(f"Similarity score: {score:.4f}")  # 4 ondalık basamak
    print("-"*50)  # Ayırıcı çizgi









    
