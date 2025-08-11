"""
9. BÖLÜM - ÖNERİ SİSTEMLERİ (RECOMMENDATION SYSTEMS) - DEEP LEARNING YAKLAŞIMI

Bu dosya Deep Learning kullanarak gelişmiş öneri sistemi oluşturur.

ÖNERİ SİSTEMLERİ NEDİR?
- Kullanıcılara kişiselleştirilmiş ürün/içerik önerileri sunar
- E-ticaret, streaming, sosyal medya platformları için kritik
- Kullanıcı deneyimini artırır, satışları yükseltir

DEEP LEARNING YAKLAŞIMI:
- Neural network tabanlı öğrenme
- Embedding layers ile kullanıcı/ürün temsilleri
- Matrix factorization ile kompleks ilişkileri öğrenir
- Non-linear etkileşimleri modelleyebilir

ÇALIŞMA PRENSİBİ:
1. Kullanıcı ve ürün ID'lerini embedding vektörlerine çevir
2. Dot product ile etkileşim skorunu hesapla
3. Dense layer ile final puanı tahmin et
4. Backpropagation ile embeddings'i optimize et

AVANTAJLARI:
✓ Karmaşık kullanıcı davranışlarını öğrenir
✓ Non-linear ilişkileri yakalar
✓ Feature engineering gerektirmez
✓ Büyük veri setlerinde etkili

DEZAVANTAJLARI:
✗ Çok veri gerektirir
✗ Hesaplama yoğun
✗ Overfitting riski
✗ Yorumlanması zor

GELENEKSEL ML İLE KARŞILAŞTIRMA:
- ML: Hızlı, basit, yorumlanabilir
- DL: Güçlü, karmaşık, veri yoğun
"""

# Deep Learning tabanlı öneri sistemi için gerekli kütüphaneler
import numpy as np                                    # Sayısal işlemler
from keras.models import Model                        # Keras model sınıfı
from keras.layers import Input, Embedding, Flatten, Dense, Dot  # Neural network katmanları
from keras.optimizers import Adam                     # Adam optimizer
from sklearn.model_selection import train_test_split  # Veri bölme
import warnings                                       # Uyarı mesajları kontrolü

# Uyarı mesajlarını gizle (temiz çıktı için)
warnings.filterwarnings("ignore")

# Örnek veri seti oluştur (küçük test verisi)
# Kullanıcı ID'leri (0-4 arası 5 farklı kullanıcı)
user_ids = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
# Ürün ID'leri (0-5 arası 6 farklı ürün)
item_ids = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 5])
# Kullanıcıların ürünlere verdiği puanlar (1-5 arası)
ratings = np.array([5, 4, 3, 2, 1, 5, 4, 3, 2, 1])

# Veriyi eğitim ve test setlerine ayır (%20 test, %80 eğitim)
(
    user_ids_train,    # Eğitim kullanıcı ID'leri
    user_ids_test,     # Test kullanıcı ID'leri
    item_ids_train,    # Eğitim ürün ID'leri
    item_ids_test,     # Test ürün ID'leri
    ratings_train,     # Eğitim puanları
    ratings_test,      # Test puanları
) = train_test_split(user_ids, item_ids, ratings, test_size=0.2, random_state=42)


# Deep Learning öneri sistemi modeli oluşturma fonksiyonu
def create_model(num_users, num_items, embedding_dim):
    # Kullanıcı girdisi için Input katmanı (tek sayı: kullanıcı ID)
    user_input = Input(shape=(1,), name="user")
    # Ürün girdisi için Input katmanı (tek sayı: ürün ID)
    item_input = Input(shape=(1,), name="item")

    # Kullanıcı ID'lerini vektörlere çeviren embedding katmanı
    user_embedding = Embedding(num_users, embedding_dim, name="user_embedding")(user_input)
    # Ürün ID'lerini vektörlere çeviren embedding katmanı
    item_embedding = Embedding(num_items, embedding_dim, name="item_embedding")(item_input)

    # Embedding çıktılarını düzleştir (2D'den 1D'ye)
    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)

    # Kullanıcı ve ürün vektörlerinin nokta çarpımını hesapla (benzerlik)
    dot_product = Dot(axes=1)([user_vec, item_vec])
    # Çıktıyı 0-1 arasına normalize et (sigmoid aktivasyon)
    output = Dense(1, activation="sigmoid")(dot_product)

    # Model oluştur (girdi: kullanıcı+ürün ID, çıktı: tahmin edilen puan)
    model = Model(inputs=[user_input, item_input], outputs=output)
    # Modeli derle (Adam optimizer, MSE loss fonksiyonu)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    return model

# Model hiperparametreleri
num_users=5        # Toplam kullanıcı sayısı
num_items=6        # Toplam ürün sayısı  
embedding_dim=8    # Embedding vektör boyutu

# Modeli oluştur
model=create_model(num_users,num_items,embedding_dim)

# Modeli eğit (100 epoch, validation split %10, verbose=1 ile ilerleme göster)
model.fit([user_ids_train,item_ids_train],ratings_train,epochs=100,validation_split=0.1,verbose=1)

# Test verisi üzerinde model performansını değerlendir
loss=model.evaluate([user_ids_test,item_ids_test],ratings_test)

# Test loss'u yazdır
print(loss)

# Örnek tahmin yap (kullanıcı 0, ürün 0 için)
user_ids=np.array([0])  # Kullanıcı ID
item_ids=np.array([0])  # Ürün ID
predictions=model.predict([user_ids,item_ids])  # Tahmin yap
# Sonucu formatla ve yazdır
print(f"predictions: {predictions[0][0]:.2f} userr_id: {user_ids[0]} item_id: {item_ids[0]}")
