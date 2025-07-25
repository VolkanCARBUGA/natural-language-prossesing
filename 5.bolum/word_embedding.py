import pandas as pd  # Veri analizi için pandas kütüphanesi
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib
from sklearn.decomposition import PCA  # Boyut azaltma için PCA algoritması
from gensim.models import Word2Vec,FastText  # Word embedding modelleri
from gensim.utils import simple_preprocess  # Metin ön işleme için basit işlemci

# Örnek cümleler - word embedding eğitimi için kullanılacak veri seti
sentences = [
    "Köpek çok tatlı bir hayvan",  # Köpek hakkında cümle
    "Kedi evcil bir hayvan",  # Kedi hakkında cümle
    "kediler genellikle bağımsız yaşarlar",  # Kedilerin davranışları
    "Köpekler  genellikle dost canlısıdır",  # Köpeklerin davranışları
    "hayvanlar insanlar için hayatın en iyi şeylerinden biridir",  # Genel hayvan sevgisi
]

# Cümleleri tokenize etme - her cümleyi kelimelerine ayırma
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Word2Vec modeli oluşturma
# vector_size=100: Her kelime için 100 boyutlu vektör
# window=5: Bağlam penceresi 5 kelime
# min_count=1: En az 1 kez geçen kelimeleri dahil et
# sg=0: CBOW algoritması kullan (1 olsa Skip-gram olurdu)
word_vectors = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)

# FastText modeli oluşturma - Word2Vec'e benzer parametreler
# FastText alt-kelime bilgisini de kullanır, bilinmeyen kelimeler için daha iyi
fasttext_model = FastText(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)

# Word embedding'leri 3 boyutlu uzayda görselleştirme fonksiyonu
def plt_word_embedding(model,title):
    word_vectors = model.wv  # Modelden kelime vektörlerini al
    words=list(word_vectors.index_to_key)  # Modeldeki tüm kelimeleri liste olarak al
    vectors=[word_vectors[word] for word in words]  # Her kelime için vektörü al
    pca= PCA(n_components=3)   # 100 boyuttan 3 boyuta indirgeme için PCA
    reduced_vectors=pca.fit_transform(vectors)  # PCA ile boyut azaltma işlemi
    fig=plt.figure(figsize=(8,6))  # 8x6 boyutunda figür oluştur
    ax=fig.add_subplot(111,projection="3d")  # 3D alt grafik ekle
    # Kelimeleri 3D uzayda nokta olarak çiz
    ax.scatter(reduced_vectors[:,0],reduced_vectors[:,1],reduced_vectors[:,2])
    # Her nokta üzerine kelimeyi yazma
    for i,word in enumerate(words):
        ax.text(reduced_vectors[i,0],reduced_vectors[i,1],reduced_vectors[i,2],word,fontsize=10)
    ax.set_xlabel("PCA 1")  # X ekseni etiketi
    ax.set_ylabel("PCA 2")  # Y ekseni etiketi
    ax.set_zlabel("PCA 3")  # Z ekseni etiketi
    ax.set_title(title)  # Grafik başlığı
    plt.show()  # Grafiği göster
    return reduced_vectors  # İndirgenen vektörleri döndür
 
# Word2Vec modelini görselleştir
plt_word_embedding(word_vectors,"Word2Vec")
# FastText modelini görselleştir
plt_word_embedding(fasttext_model,"FastText")
    
    
       






