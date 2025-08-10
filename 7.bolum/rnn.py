# === KÜTÜPHANE İMPORTLARI ===
import numpy as np  # Sayısal hesaplamalar ve array manipülasyonu için NumPy kütüphanesi
import pandas as pd  # Veri okuma, manipülasyon ve analizi için Pandas kütüphanesi
from gensim.models import Word2Vec  # Kelimeler için vector representation oluşturan Word2Vec modeli
from keras_preprocessing.sequence import pad_sequences  # Farklı uzunluktaki text dizilerini aynı boyuta getirmek için
from keras.models import Sequential  # Katmanları sıralı olarak ekleyebileceğimiz Keras model tipi
from keras.layers import SimpleRNN, Dense, Embedding, Dropout  # RNN, dense, embedding ve dropout katmanları
from keras_preprocessing.text import Tokenizer  # Metinleri sayısal token dizilerine dönüştüren sınıf
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine bölen fonksiyon
from sklearn.preprocessing import LabelEncoder  # String etiketleri sayısal değerlere çeviren sınıf

# === VERİ SETİ OLUŞTURMA ===
# Restoran yorumları ve sentiment analizi için kullanılacak Türkçe veri seti
# Her yorum positive (pozitif) veya negative (negatif) olarak etiketlenmiş
data = {
    "text": [  # Restoran deneyimleri hakkında gerçek Türkçe yorumlar
        "Yemekler mükemmeldi, kesinlikle tavsiye ederim.",  # Pozitif yorum - yemek kalitesi
        "Servis çok yavaştı, saatlerce bekledik.",  # Negatif yorum
        "Garsonlar çok nazikti ve güler yüzlüydü.",  # Pozitif yorum
        "Tatlılar bayattı, hiç beğenmedik.",  # Negatif yorum
        "Atmosfer çok hoştu, arkadaşlarla keyifli zaman geçirdik.",  # Pozitif yorum
        "Yemekler çok soğuktu, hayal kırıklığına uğradım.",  # Negatif yorum
        "Fiyat-performans açısından gayet başarılı.",  # Pozitif yorum
        "Mekân çok kalabalıktı ve gürültülüydü.",  # Negatif yorum
        "Etin pişme derecesi tam istediğim gibiydi.",  # Pozitif yorum
        "Menüdeki çoğu ürün mevcut değildi.",  # Negatif yorum
        "Sunum çok şıktı ve özenliydi.",  # Pozitif yorum
        "Yemeklerde tuz oranı fazlaydı.",  # Negatif yorum
        "Tatlılar enfesti, özellikle cheesecake harikaydı.",  # Pozitif yorum
        "Masamızın yanındaki çocuklar çok gürültü yapıyordu.",  # Negatif yorum
        "Rezervasyon sistemi çok kolaydı ve hızlıydı.",  # Pozitif yorum
        "Siparişim yanlış geldi, değişimi de çok sürdü.",  # Negatif yorum
        "Garsonlar çok ilgiliydi, sürekli bizimle ilgilendiler.",  # Pozitif yorum
        "İçecekler çok geç geldi.",  # Negatif yorum
        "Yemeklerin sunumu göze hitap ediyordu.",  # Pozitif yorum
        "Pilav kuru ve tatsızdı.",  # Negatif yorum
        "Şef bizimle ilgilendi, çok mutlu olduk.",  # Pozitif yorum
        "Salata taze değildi, bayat marullar vardı.",  # Negatif yorum
        "Yemeklerin porsiyonları çok büyüktü.",  # Pozitif yorum
        "Tatlıdan plastik tadı geldi.",  # Negatif yorum
        "İçeride hoş bir müzik çalıyordu.",  # Pozitif yorum
        "Masamız yağlıydı, silinmemişti.",  # Negatif yorum
        "Servis hızlıydı ve siparişimiz doğru geldi.",  # Pozitif yorum
        "Sandalyeler rahatsızdı.",  # Negatif yorum
        "Garson menüyü çok iyi tanıtıyordu.",  # Pozitif yorum
        "Yemekler birbirine karışmış halde geldi.",  # Negatif yorum
        "Harika bir deneyimdi, tekrar geleceğim.",  # Pozitif yorum
        "Çalışanlar sürekli göz teması kuruyordu, çok profesyoneldi.",  # Pozitif yorum
        "Tatlıların tadı eksikti.",  # Negatif yorum
        "Ana yemek lezzetliydi ama fiyatı yüksekti.",  # Negatif yorum
        "İkram olarak gelen çay çok güzeldi.",  # Pozitif yorum
        "Masada sinek vardı.",  # Negatif yorum
        "Servis elemanları kibar ve ilgiliydi.",  # Pozitif yorum
        "Yemeklerde yağ oranı çok fazlaydı.",  # Negatif yorum
        "Dekorasyon harika, çok ferah bir ortam.",  # Pozitif yorum
        "Lavabolar çok pisti.",  # Negatif yorum
        "Fırından yeni çıkmış ekmekler harikaydı.",  # Pozitif yorum
        "Koltuklar kirliydi.",  # Negatif yorum
        "Yemek sıcaktı ve taze geldi.",  # Pozitif yorum
        "Garson siparişleri karıştırdı.",  # Negatif yorum
        "Çorba tam kıvamındaydı.",  # Pozitif yorum
        "Kahve yanık kokuyordu.",  # Negatif yorum
        "Mekânın ambiyansı çok hoştu.",  # Pozitif yorum
        "Menüdeki çeşitlilik yeterli değildi.",  # Negatif yorum
        "Tatlı sunumu çok estetikti.",  # Pozitif yorum
        "Kullanılan malzemeler kalitesizdi.",  # Negatif yorum
        "Mutfaktan gelen koku rahatsız ediciydi.",  # Negatif yorum
        "Fiyatlar gayet makul.",  # Pozitif yorum
        "Yemek beklediğimizden hızlı geldi.",  # Pozitif yorum
        "Sandalyelerde oturmak çok rahattı.",  # Pozitif yorum
        "Çalışanlar agresifti.",  # Negatif yorum
        "Çocuk menüsü çok iyi düşünülmüş.",  # Pozitif yorum
        "İçecekler buzsuz geldi.",  # Negatif yorum
        "Kebaplar tam kıvamında pişmişti.",  # Pozitif yorum
        "Salata sosu çok ekşiydi.",  # Negatif yorum
        "Mekân çok şık dekore edilmişti.",  # Pozitif yorum
        "Çorba çok tuzluydu.",  # Negatif yorum
        "Tatlılar aşırı şekerliydi.",  # Negatif yorum
        "Yemek servisi çok düzenliydi.",  # Pozitif yorum
        "Rezervasyonumuz olmasına rağmen 20 dakika bekledik.",  # Negatif yorum
        "Tatlılar çok özenli hazırlanmıştı.",  # Pozitif yorum
        "Garsonlar sert davranıyordu.",  # Negatif yorum
        "İçeri girer girmez sıcak karşılama aldık.",  # Pozitif yorum
        "Kötü bir deneyimdi, tavsiye etmem.",  # Negatif yorum
        "Servis çok nazikti, kendimi özel hissettim.",  # Pozitif yorum
        "Masalar arası mesafe çok iyiydi, ferah ortam.",  # Pozitif yorum
        "Yemekler çok hızlı geldi, sıcaktı.",  # Pozitif yorum
        "Menüde vegan seçeneklerin olması çok hoşuma gitti.",  # Pozitif yorum
        "Yemekler güzel ama servis biraz daha iyi olabilirdi.",  # Negatif yorum
        "Tatlı çok sertti, zor yedim.",  # Negatif yorum
        "Garsonlar çok yorgun görünüyordu, ilgisizdi.",  # Negatif yorum
        "İç mekan tasarımı etkileyiciydi.",  # Pozitif yorum
        "Tatlıların porsiyonu küçüktü ama lezzetliydi.",  # Pozitif yorum
        "Girişte karşılama çok iyiydi, detaylara önem verilmiş.",  # Pozitif yorum
        "Baharatlar çok yoğundu, tadı bastırıyordu.",  # Negatif yorum
        "Menüdeki her şey açık ve netti.",  # Pozitif yorum
        "Garson çok kaba davrandı.",  # Negatif yorum
        "Koltuklar çok rahattı, saatlerce oturduk.",  # Pozitif yorum
        "Etler çok iyi marine edilmişti.",  # Pozitif yorum
        "İkram tatlı güzel bir sürprizdi.",  # Pozitif yorum
        "Yemeklerde kullanılan yağ midemi rahatsız etti.",  # Negatif yorum
        "Tatlı çok sıradandı, özel bir şey değildi.",  # Negatif yorum
        "Kahve sunumu çok şıktı.",  # Pozitif yorum
        "Garsonlar siparişimizi unuttu.",  # Negatif yorum
        "Lavabolar çok temizdi.",  # Pozitif yorum
        "Garson menüyü ezbere biliyordu.",  # Pozitif yorum
        "Etin içi çiğdi, pişmemişti.",  # Negatif yorum
        "Masaya servis yapan kişi çok kibardı.",  # Pozitif yorum
        "Yemek sonrası gelen lokum hoştu.",  # Pozitif yorum
        "Yemekte çok fazla soğan vardı.",  # Negatif yorum
        "Yemek sonrası çay ikramı çok güzeldi.",  # Pozitif yorum
        "Kahvaltı tabağı çok doyurucuydu.",  # Pozitif yorum
        "Yemekler çok baharatlıydı, yiyemedim.",  # Negatif yorum
        "Tatlılar taze yapılmıştı, sıcaktı.",  # Pozitif yorum
        "Garson yanlış fatura getirdi.",  # Negatif yorum
        "Mekânda sigara kokusu vardı.",  # Negatif yorum

    ],
    "label": [  # Her yorumun duygu durumu etiketi (positive=pozitif, negative=negatif)
        "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive", "negative", "positive", "positive",
        "negative", "negative", "positive", "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "negative", "positive", "positive", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "positive", "positive", "positive", "negative", "positive",
        "negative", "positive", "positive", "positive", "negative", "negative", "positive", "negative",
        "positive", "positive", "negative", "positive", "positive", "negative", "positive", "negative",
        "negative", "positive", "positive", "negative",
        ]
}

# === VERİ SETİ KONTROLÜ VE DATAFRAME OLUŞTURMA ===
print("Toplam yorum sayısı:", len(data["text"]))     # Text veri sayısını kontrol et
print("Toplam etiket sayısı:", len(data["label"]))   # Label sayısını kontrol et
df = pd.DataFrame(data)  # Dictionary'yi pandas DataFrame'e çevir (daha kolay manipülasyon için)

# === METİN TOKENİZASYONU ===
# Metinleri RNN modelinin anlayabileceği sayısal forma çevirme işlemi
tekonizer = Tokenizer()  # Metinleri tokenlara çevirecek nesne oluştur
tekonizer.fit_on_texts(df["text"])  # Tüm metinler üzerinde kelime sözlüğü (vocabulary) oluştur
sequences = tekonizer.texts_to_sequences(df["text"])  # Her metni sayısal token dizisine çevir
word_index = tekonizer.word_index  # Kelime → sayı eşleştirmesini al (örn: {"yemek": 1, "güzel": 2})

# === SEQUENCE PADDING İŞLEMİ ===
# RNN'ler sabit uzunlukta girdi bekler, bu yüzden tüm dizileri aynı boyuta getirmemiz gerekir
maxlen = max(len(sequence) for sequence in sequences)  # Veri setindeki en uzun cümlenin token sayısını bul
X = pad_sequences(sequences, maxlen=maxlen)  # Kısa cümlelerin başına 0 ekleyerek tümünü aynı uzunluğa getir
print("X shape (girdi matrisi boyutu):", X.shape)  # Girdi matrisinin boyutunu kontrol et [sample_sayısı, max_uzunluk]

# === ETİKET ENCODING ===
# String etiketleri (positive/negative) sayısal değerlere (1/0) çevirme işlemi
label_encoder = LabelEncoder()  # String → sayı dönüştürücü nesne oluştur
y = label_encoder.fit_transform(df["label"])  # "positive"→1, "negative"→0 dönüşümü yap
print("y unique values (etiket dağılımı):", np.unique(y, return_counts=True))  # Kaç tane pozitif/negatif olduğunu göster

# === VERİ SETİNİ AYIRMA ===
# Modelin performansını objektif olarak değerlendirmek için veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                    # Girdi ve hedef veriler
    test_size=0.2,          # %20'si test için ayrıl
    random_state=42         # Tekrarlanabilir sonuçlar için sabit random seed
)

# === WORD2VEC EMBEDDİNG MODELİ OLUŞTURMA ===
# Kelimeleri dense vector'lere çevirmek için Word2Vec modeli eğit
sentences = [text.split() for text in df["text"]]  # Her metni kelime listesine çevir (Word2Vec formatı için)
word2vec_model = Word2Vec(
    sentences,           # Eğitim verileri (kelime listeleri)
    vector_size=100,     # Her kelime 100 boyutlu vektörle temsil edilecek
    window=5,           # Context window: bir kelimenin etrafındaki 5 kelimeyi dikkate al
    min_count=1,        # En az 1 kez geçen kelimeler dahil edilsin
    sg=1               # Skip-gram algoritması kullan (CBOW yerine)
)

# === EMBEDDİNG MATRİSİ OLUŞTURMA ===
# Word2Vec'ten öğrenilen vektörleri Keras embedding katmanında kullanmak için matris oluştur
embedding_dim = 100  # Her kelime 100 boyutlu vektörle temsil edilecek
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))  # Boş embedding matrisi oluştur (kelime_sayısı × 100)

# Tokenizer'daki her kelime için Word2Vec vektörünü al
for word, i in word_index.items():  # word_index'teki her kelime-indeks çifti için
    if word in word2vec_model.wv:  # Eğer kelime Word2Vec modelinde mevcutsa
        embedding_matrix[i] = word2vec_model.wv[word]  # O kelimenin vektörünü embedding matrisine koy

# === RNN MODELİ OLUŞTURMA ===
model = Sequential()  # Katmanları sıralı olarak ekleyeceğimiz model oluştur

# 1. Embedding Katmanı - Token'ları yoğun vektörlere çevirir
model.add(
    Embedding(
        input_dim=len(word_index) + 1,      # Giriş boyutu: toplam kelime sayısı + 1 (padding için)
        output_dim=embedding_dim,           # Çıkış boyutu: her kelime 100 boyutlu vektöre çevrilir
        weights=[embedding_matrix],         # Önceden eğitilmiş Word2Vec ağırlıklarını kullan
        input_length=maxlen,               # Giriş sequence'inin uzunluğu (padding sonrası sabit)
        trainable=False,                   # Embedding ağırlıklarını dondurun (sadece RNN eğitilsin)
    )
)

# 2. SimpleRNN Katmanı - Temel tekrarlayan sinir ağı katmanı
model.add(SimpleRNN(
    units=128,                  # RNN katmanındaki gizli nöron sayısı
    return_sequences=False,     # Sadece son time step'in çıktısını döndür (sequence-to-one)
    dropout=0.2,               # Input bağlantılarında %20 dropout (overfitting önleme)
    recurrent_dropout=0.2      # Tekrarlayan (hidden state) bağlantılarda %20 dropout
))

# 3. Regularization ve Dense Katmanlar
model.add(Dropout(0.5))  # %50 dropout katmanı - güçlü regularization
model.add(Dense(64, activation="relu"))  # 64 nöronlu ara katman (ReLU aktivasyon fonksiyonu)
model.add(Dropout(0.3))  # %30 dropout katmanı - orta seviye regularization  
model.add(Dense(1, activation="sigmoid"))  # Çıktı katmanı: 1 nöron + sigmoid (binary classification için)

# === MODEL DERLEME ===
# Modelin eğitim stratejisini belirleme (optimizer, loss function, metrikler)
model.compile(
    optimizer="adam",                    # Adam optimizatörü: adaptif learning rate ile hızlı konverjans
    loss="binary_crossentropy",         # Binary crossentropy: ikili sınıflandırma için uygun loss function
    metrics=["accuracy"]                # Accuracy: modelin başarı oranını takip etmek için
)

# === MODEL EĞİTİMİ ===
print("Model eğitimi başlıyor...")
history = model.fit(
    X_train, y_train,                   # Eğitim verisi (girdi ve hedef)
    epochs=20,                          # 20 epoch boyunca eğitim (veri setini 20 kez geç)
    batch_size=32,                      # Her iterasyonda 32 örnek birden işle
    validation_data=(X_test, y_test),   # Test verisini validation için kullan (overfitting kontrolü)
    verbose=1                           # Eğitim sürecini detaylı ekranda göster
)

# === MODEL DEĞERLENDİRME ===
# Eğitilmiş modelin test seti üzerindeki performansını ölç
loss, accuracy = model.evaluate(X_test, y_test)  # Test setinde loss ve accuracy hesapla
print(f"Test Loss (test kaybı): {loss:.4f}")     # Test loss değerini 4 ondalık basamakla yazdır
print(f"Test Accuracy (test doğruluğu): {accuracy:.4f}")  # Test accuracy değerini 4 ondalık basamakla yazdır

# === YENİ METİNLER İÇİN TAHMİN FONKSİYONU ===
def classify_sentences(sentences):
    """
    Yeni metinlerin sentiment'ini (pozitif/negatif) tahmin eden fonksiyon
    
    Args:
        sentences (list): Tahmin edilecek metin listesi
    
    Returns:
        list: Her metin için "positive" veya "negative" etiketleri
    """
    # 1. Metinleri modelin anlayabileceği formata çevir
    sequences = tekonizer.texts_to_sequences(sentences)  # Metinleri sayısal token dizilerine çevir
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)  # Eğitim verisindeki uzunluğa göre padding uygula
    
    # 2. Model ile tahmin yap
    predictions = model.predict(padded_sequences)  # Her metin için 0-1 arası olasılık değeri al
    predicted_classes = (predictions > 0.5).astype(int).flatten()  # 0.5 threshold: >0.5 ise pozitif (1), <0.5 ise negatif (0)
    
    # 3. Sayısal sonuçları text etiketlere çevir
    labels = ["positive" if pred == 1 else "negative" for pred in predicted_classes]  # 1→positive, 0→negative
    return labels

# === MODEL TESTİ ===
# Eğitilmiş modeli gerçek bir cümle ile test etme
sentence = "garson çok iyiydi"  # Test edilecek Türkçe cümle (pozitif bir yorum)
predictions = classify_sentences([sentence])  # Fonksiyonu çağırarak sentiment tahmini yap
print(f"Cümle: '{sentence}'")  # Test edilen cümleyi yazdır
print(f"Tahmin: {predictions[0]}")  # Modelin tahmini sonucunu yazdır (positive/negative)


"""
=== RNN (Recurrent Neural Network) Nedir? ===

RNN (Tekrarlayan Sinir Ağları), özellikle sıralı veriler (sequential data) için tasarlanmış 
bir yapay sinir ağı türüdür. Bu kodda doğal dil işleme (NLP) alanında sentiment analizi 
(duygu analizi) için kullanılmıştır.

=== RNN'in Temel Özellikleri ===

1. **Hafıza (Memory)**: RNN'ler önceki adımlardan gelen bilgiyi hatırlayabilir
2. **Sıralı İşleme**: Veriyi adım adım, sırasıyla işler
3. **Değişken Uzunluk**: Farklı uzunluktaki girdi dizilerini işleyebilir
4. **Tekrarlayan Yapı**: Aynı ağırlıklar her zaman adımında kullanılır

=== Bu Kodda RNN Nasıl Kullanılıyor? ===

1. **Veri Hazırlama**:
   - Türkçe restoran yorumları (text) ve sentiment etiketleri (positive/negative)
   - Metinler tokenize edilip sayısal dizilere çevriliyor
   - Tüm diziler aynı uzunluğa getiriliyor (padding)

2. **Word Embedding**:
   - Word2Vec ile kelimeler 100 boyutlu vektörlere çevriliyor
   - Bu vektörler kelimelerin anlamsal ilişkilerini yakalar

3. **Model Mimarisi**:
   - Embedding Layer: Kelimeleri vektörlere çevirir
   - SimpleRNN Layer: Sıralı bilgiyi işler (128 nöron)
   - Dense Layers: Sınıflandırma için tam bağlı katmanlar
   - Dropout: Overfitting'i önler

4. **Eğitim**:
   - Binary crossentropy loss ile ikili sınıflandırma
   - Adam optimizer ile ağırlık güncellemesi
   - 20 epoch boyunca eğitim

=== RNN'in Avantajları ===

- Cümledeki kelime sırasını ve bağlamı anlayabilir
- "çok iyi" ile "iyi çok" arasındaki farkı algılayabilir
- Değişken uzunluktaki metinleri işleyebilir
- Önceki kelimelerin etkisini sonraki tahminlerde kullanır

=== RNN'in Dezavantajları ===

- Uzun dizilerde gradient vanishing problemi
- LSTM ve GRU gibi gelişmiş versiyonlar genelde daha başarılı
- Paralel işleme konusunda sınırlı

=== Sentiment Analizi Uygulaması ===

Bu kod, restoran yorumlarının pozitif veya negatif olduğunu tahmin ediyor:
- "garsonlar çok iyiydi" → Pozitif
- "servis çok yavaştı" → Negatif

Model, cümledeki kelimelerin sırasını ve birbirleriyle olan ilişkilerini
öğrenerek doğru sentiment tahmini yapabiliyor.

=== GELECEK GELİŞTİRMELER ===

**Model İyileştirmeleri:**
- LSTM veya GRU kullanarak hafıza kapasitesini artırma
- Bidirectional RNN ile hem geçmiş hem gelecek context'i kullanma
- Attention mechanism ekleme
- Ensemble methods ile farklı modelleri birleştirme

**Veri Augmentation:**
- Daha fazla restoran yorumu ekleme
- Farklı domain'lerden (otel, film, ürün) yorumlar
- Data balancing teknikleri
- Synthetic data generation

**Hyperparameter Tuning:**
- Learning rate scheduling
- Farklı optimizer'lar (RMSprop, SGD)
- Model architecture experiments
- Cross-validation ile robust evaluation

Bu RNN implementasyonu, NLP pipeline'ının tüm adımlarını içermekte ve 
gerçek dünya sentiment analysis uygulamaları için sağlam bir temel oluşturmaktadır.
"""




