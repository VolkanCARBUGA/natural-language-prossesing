"""
9. BÖLÜM - ÖNERİ SİSTEMLERİ (RECOMMENDATION SYSTEMS) - GELENEKSEL ML YAKLAŞIMI

Bu dosya geleneksel makine öğrenmesi ile öneri sistemi oluşturur.

COLLABORATIVE FILTERING NEDİR?
- Kullanıcıların geçmiş davranışlarına dayalı öneriler
- "Benzer kullanıcılar benzer ürünleri sever" mantığı
- İki ana yaklaşım: User-based ve Item-based

K-NEAREST NEIGHBORS (KNN) YAKLAŞIMI:
- Benzer kullanıcıları bulur (cosine similarity ile)
- Komşu kullanıcıların tercihlerini analiz eder
- En yakın K komşunun ortalama puanını hesaplar

AVANTAJLARI:
✓ Hızlı ve etkili
✓ Yorumlanabilir sonuçlar
✓ Az hesaplama kaynağı gerektirir
✓ Basit implementasyon

DEZAVANTAJLARI:
✗ Cold start problemi (yeni kullanıcı/ürün)
✗ Sparse data problemi
✗ Scalability sorunları
✗ Popularity bias

MOVIELENS VERİ SETİ:
- 100,000 film puanlaması
- 943 kullanıcı, 1682 film
- 1-5 puan skalası
- GroupLens Research tarafından sağlanan

KULLANIM ALANLARI:
- E-ticaret platformları
- Streaming servisleri (Netflix, Spotify)
- Sosyal medya önerileri
- Haber/makale önerileri
"""

# Scikit-Surprise kütüphanesi - geleneksel makine öğrenmesi tabanlı öneri sistemi
from surprise import Dataset,KNNBasic,accuracy  # Öneri sistemi algoritmaları
from surprise.model_selection import train_test_split  # Veri bölme

# MovieLens 100k veri setini yükle (film puanlama verileri)
# Bu veri seti 100,000 film puanlaması içerir (kullanıcı-film-puan üçlüleri)
data=Dataset.load_builtin("ml-100k")

# Veriyi eğitim (%80) ve test (%20) setlerine ayır
trainset,testset=train_test_split(data,test_size=0.2,random_state=42)

# KNN model seçenekleri: cosine similarity ve user-based filtering
model_options={"name":"cosine","user_based":True}

# K-Nearest Neighbors tabanlı öneri sistemi oluştur
model=KNNBasic(sim_options=model_options)
# Modeli eğitim verisi ile eğit
model.fit(trainset)
# Test verisi üzerinde tahminler yap
predictions=model.test(testset)
# Root Mean Square Error (RMSE) ile model performansını değerlendir
accuracy.rmse(predictions)

# Her kullanıcı için en iyi N öneriyi bulan fonksiyon
def get_top_n(predictions,n=10):
    # Kullanıcı bazında önerileri saklayacak sözlük
    top_n={}
    # Her tahmin için (kullanıcı, ürün, gerçek_puan, tahmin_edilen_puan, _)
    for uid,iid,true_r,est,_ in predictions:
        # Kullanıcı sözlükte yoksa boş liste oluştur
        if uid not in top_n:
            top_n[uid]=[]
        # Kullanıcının listesine (ürün_id, tahmin_puanı) ekle
        top_n[uid].append((iid,est))
    
    # Her kullanıcı için önerileri puana göre sırala ve en iyi N'i al
    for uid,user_ratings in top_n.items():
        # Tahmin puanına göre büyükten küçüğe sırala
        user_ratings.sort(key=lambda x:x[1],reverse=True)
        # İlk N tanesini al
        top_n[uid]=user_ratings[:n]
    return top_n

# En iyi 5 öneriyi al
n=5
top_n=get_top_n(predictions,n)
# Kullanıcı 100 için önerileri göster
user_id="100"
print(f"Top {n} recommendations for user {user_id}")
# Her öneriyi yazdır (ürün ID'si ve tahmin puanı ile)
for item_id,rating in top_n[user_id]:
    print(f"Item {item_id}: Score {rating:.2f}")










