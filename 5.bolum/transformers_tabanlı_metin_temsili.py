# Transformers kütüphanesinden gerekli sınıfları import ediyoruz
from transformers import AutoTokenizer,AutoModel  # BERT modelini yüklemek için gerekli sınıflar
import torch  # PyTorch tensor işlemleri için
import numpy as np  # Numpy array işlemleri için

# Kullanılacak BERT modelinin adını belirliyoruz
model_name="bert-base-uncased"  # Küçük harfli, temel BERT modeli

# Tokenizer'ı yüklüyoruz - metni sayısal tokens'lara çevirir
tokenizer=AutoTokenizer.from_pretrained(model_name)

# BERT modelini yüklüyoruz - gerçek dil modelimiz
model=AutoModel.from_pretrained(model_name)

# İşleyeceğimiz örnek metin
text="Hello, how are you?"  # Basit bir İngilizce cümle

# Metni tokenizer ile işliyoruz ve PyTorch tensörü olarak dönüştürüyoruz
inputs=tokenizer(text,return_tensors="pt")  # "pt" = PyTorch tensörleri

# Gradyan hesaplamayı kapatıyoruz (inference için gerekli değil, hızlandırır)
with torch.no_grad():
    # Modelden çıktıları alıyoruz
    outputs=model(**inputs)  # BERT modelinden geçiriyoruz
    
# Son gizli katman durumlarını alıyoruz (her token için embedding vektörleri)
last_hidden_state=outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

# İlk token'ın embedding vektörünü numpy array'e çeviriyoruz
first_token_embedding=last_hidden_state[0,0:].numpy()  # İlk cümlenin ilk token'ı ([CLS] token'ı)

# İlk token embedding'inin boyutunu yazdırıyoruz
print("First token embedding shape:",first_token_embedding)  # BERT-base için 768 boyutlu olmalı


"""
=== TRANSFORMERS TABANLI METİN TEMSİLİ AÇIKLAMASI ===

Bu kod, modern doğal dil işlemede devrim yaratan BERT modelini kullanarak 
metin temsillerini nasıl elde edeceğimizi göstermektedir.

1. BERT (Bidirectional Encoder Representations from Transformers):
   - Google tarafından geliştirilen çift yönlü dil modeli
   - Geleneksel yöntemlerden farklı olarak hem sağdan hem soldan context anlayabilir
   - Transformer mimarisi üzerine kurulu

2. Tokenization Süreci:
   - Metin önce alt-kelime (subword) parçalarına bölünür
   - Her parça bir sayısal ID'ye dönüştürülür  
   - Özel tokenlar eklenir: [CLS] (başlangıç), [SEP] (ayırıcı), [PAD] (dolgu)

3. Embedding Süreci:
   - Her token 768 boyutlu bir vektöre dönüştürülür (BERT-base için)
   - Bu vektörler kelimenin anlamını ve context'ini içerir
   - last_hidden_state her tokenin final embedding'ini verir

4. [CLS] Token'ın Önemi:
   - İlk token olan [CLS], tüm cümlenin özetini içerir
   - Sınıflandırma görevlerinde en çok kullanılan representation
   - Cümle düzeyinde anlamsal bilgiyi barındırır

5. Geleneksel Yöntemlerle Karşılaştırma:
   - TF-IDF/Bag of Words: Sadece kelime sıklığı, context yok
   - Word2Vec: Statik embeddings, context'e göre değişmez  
   - BERT: Dinamik embeddings, her cümlede farklı representation

6. Pratik Kullanım Alanları:
   - Metin sınıflandırma (sentiment analysis, spam detection)
   - Soru-cevap sistemleri
   - Metin benzerlik hesaplama
   - Dil çevirisi ve özetleme

Bu yaklaşım, metinlerin daha zengin ve anlamlı temsillerini elde etmemizi sağlar,
bu da NLP görevlerinde çok daha iyi performans verir.
"""






