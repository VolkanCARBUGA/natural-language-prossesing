# Gerekli kütüphaneleri içe aktarma
import pandas as pd  # Veri analizi ve DataFrame işlemleri için
import matplotlib.pyplot as plt  # Grafik çizimi için
from sklearn.decomposition import PCA  # Boyut azaltma için (Principal Component Analysis)
from sklearn.cluster import KMeans  # K-Means kümeleme algoritması için
from gensim.models import Word2Vec,FastText  # Word embedding modelleri için
from gensim.utils import simple_preprocess  # Metin ön işleme için
import re  # Düzenli ifadeler (regex) için


df=pd.read_csv("5.bolum/IMDB Dataset.csv")  # IMDB film yorumları veri setini okuma

documents=df["review"]  # Sadece 'review' sütununu alarak yorumları documents değişkenine atama

# Metin temizleme fonksiyonu tanımlama
def clean_text(text):
    text=text.lower()  # Tüm harfleri küçük harfe çevirme
    text=re.sub(r"\d+"," ",text)  # Sayıları boşlukla değiştirme
    text=re.sub(r"[^\w\s]"," ",text)  # Özel karakterleri (noktalama işaretleri) boşlukla değiştirme
    text=" ".join(word for word in text.split() if len(word)>3)  # 3 karakterden uzun kelimeleri seçme
    text=text.strip()  # Başta ve sondaki boşlukları temizleme
    return text  # Temizlenmiş metni döndürme




cleaned_documents=[clean_text(doc) for doc in documents]  # Tüm belgeleri temizleme

tokenized_documents=[simple_preprocess(doc) for doc in cleaned_documents]  # Belgeleri token'lara ayırma (kelime listelerine dönüştürme)

# Word2Vec modelini eğitme
word2vec_model=Word2Vec(tokenized_documents,  # Eğitim verisi
                       vector_size=100,  # Her kelime için 100 boyutlu vektör
                       window=5,  # Bağlam penceresi büyüklüğü (5 kelime öncesi ve sonrası)
                       min_count=1,  # Minimum kelime frekansı (1 kez geçen kelimeler dahil)
                       sg=0)  # CBOW algoritması kullanma (sg=1 olsaydı Skip-Gram olurdu)

word_vectors=word2vec_model.wv  # Eğitilmiş kelime vektörlerini alma
words=list(word_vectors.index_to_key)[:500]  # İlk 500 kelimeyi seçme (görselleştirme için)

vectors=[word_vectors[word] for word in words]  # Seçilen kelimelerin vektörlerini alma

# K-Means kümeleme algoritması uygulama
kmeans=KMeans(n_clusters=5,random_state=42)  # 5 küme oluşturma, rastgelelik için seed=42
kmeans.fit(vectors)  # Vektörleri kümeleme
cluster_labels=kmeans.labels_   # Her vektörün hangi kümeye ait olduğunu alma

# PCA ile boyut azaltma (100 boyuttan 2 boyuta)
pca=PCA(n_components=2)  # 2D görselleştirme için 2 bileşen
reduced_vectors=pca.fit_transform(vectors)  # Vektörleri 2D'ye dönüştürme

# Görselleştirme başlatma
plt.figure()  # Yeni bir grafik penceresi oluşturma
plt.scatter(reduced_vectors[:,0],  # X ekseni (1. PCA bileşeni)
           reduced_vectors[:,1],  # Y ekseni (2. PCA bileşeni)
           c=cluster_labels,  # Renkleri küme etiketlerine göre ayarlama
           cmap="viridis")  # Renk paleti olarak viridis kullanma

# Küme merkezlerini görselleştirme
centers=pca.transform(kmeans.cluster_centers_)  # Küme merkezlerini de 2D'ye dönüştürme
plt.scatter(centers[:,0],centers[:,1],  # Küme merkezlerinin koordinatları
           c="red",s=100,alpha=0.75,marker="x",label="Centroids")  # Kırmızı X işareti olarak gösterme

# Her kelimenin grafikteki konumuna etiket ekleme
for i,word in enumerate(words):
    plt.text(reduced_vectors[i,0],reduced_vectors[i,1],word,fontsize=8)  # Kelimeyi koordinatına yazma

# Grafik başlık ve eksen etiketleri
plt.title("Word Embedding K-Means Clustering")  # Grafik başlığı
plt.xlabel("PCA 1")  # X ekseni etiketi
plt.ylabel("PCA 2")  # Y ekseni etiketi
plt.show()  # Grafiği gösterme


"""
WORD EMBEDDING VE K-MEANS KÜMELEMESİ AÇIKLAMASI:

Bu kod, Doğal Dil İşleme (NLP) alanında çok önemli bir konuyu ele almaktadır: Word Embedding ve Kümeleme.

1. WORD EMBEDDING NEDİR?
   - Kelimeleri sayısal vektörlerle temsil etme yöntemidir
   - Bu kodda Word2Vec algoritması kullanılmıştır
   - Her kelime 100 boyutlu bir vektörle temsil edilir
   - Anlamsal olarak benzer kelimeler, vektör uzayında birbirine yakın konumlanır

2. WORD2VEC ALGORİTMASI:
   - CBOW (Continuous Bag of Words) modeli kullanılmıştır (sg=0)
   - Bağlam kelimeleri kullanarak hedef kelimeyi tahmin eder
   - Window=5: Her kelime için 5 kelime öncesi ve sonrasını bağlam olarak kullanır
   - Vector_size=100: Her kelime 100 boyutlu vektörle temsil edilir

3. K-MEANS KÜMELEMESİ:
   - Kelime vektörlerini 5 gruba ayırmak için kullanılmıştır
   - Benzer anlamlı kelimeler aynı kümeye düşer
   - Örneğin: pozitif duygular, negatif duygular, nesneler vb. ayrı kümelerde olabilir

4. BOYUT AZALTMA (PCA):
   - 100 boyutlu vektörler görselleştirme için 2 boyuta indirilmiştir
   - PCA, varyansı maksimum yapan 2 ana bileşeni bulur
   - Bu sayede yüksek boyutlu veriyi 2D grafikte gösterebiliriz

5. GÖRSELLEŞTİRME:
   - Her nokta bir kelimeyi temsil eder
   - Renkler küme aidiyetini gösterir
   - Kırmızı X işaretleri küme merkezlerini gösterir
   - Yakın kelimeler grafik üzerinde birbirine yakın görünür

6. UYGULAMA ALANLARI:
   - Sentiment analizi
   - Belge sınıflandırma
   - Makine çevirisi
   - Arama motorları
   - Öneri sistemleri

Bu yöntem, kelimelerin anlamsal ilişkilerini matematiksel olarak modellememizi sağlar
ve NLP uygulamalarının temelini oluşturur.
"""









