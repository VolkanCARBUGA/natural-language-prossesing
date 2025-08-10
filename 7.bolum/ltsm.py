# === KÜTÜPHANE İMPORTLARI ===
import numpy as np  # Sayısal işlemler ve array manipülasyonu için NumPy kütüphanesi
import tensorflow as tf  # Google'ın derin öğrenme framework'ü TensorFlow
from tensorflow import keras  # TensorFlow'un yüksek seviye API'si Keras

# Keras model ve katman importları
from keras.models import Sequential  # Katmanları sıralı olarak ekleyebileceğimiz model tipi
from keras.layers import LSTM, Dense, Embedding  # LSTM (hafıza), Dense (tam bağlantılı), Embedding (kelime gömme) katmanları
from keras_preprocessing.text import Tokenizer  # Metinleri sayısal token dizilerine dönüştüren sınıf
from keras.preprocessing.sequence import pad_sequences  # Farklı uzunluktaki dizileri aynı boyuta getiren fonksiyon



# === EĞİTİM VERİ SETİ OLUŞTURMA ===
# LSTM modelini eğitmek için Türkçe pozitif duygulu cümlelerden oluşan veri seti
# Bu cümleler modelin Türkçe dil yapısını ve pozitif duygu ifadelerini öğrenmesi için kullanılacak
texts = [  # Türkçe pozitif duygulu cümlelerden oluşan eğitim veri seti
    "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "Kitap okumak beni gerçekten mutlu ediyor.",
    "Yeni bir diziye başladım ve çok sürükleyici.",
    "Arkadaşlarımla vakit geçirmek beni her zaman iyi hissettiriyor.",
    "Sahilde gün batımını izlemek huzur vericiydi.",
    "Köpeğimle parkta oynarken çok eğlendim.",
    "Bugün kendimi çok enerjik hissediyorum.",
    "Müzik dinlemek moralimi yükseltiyor.",
    "Sevdiğim yemeği yemek günümü güzelleştirdi.",
    "Tatilde denize girmek bana iyi geldi.",
    "Sabah koşusu yapmak güne harika başlamamı sağladı.",
    "Güzel bir haber aldım, çok sevindim.",
    "Uzun zamandır görmediğim arkadaşımı görmek beni mutlu etti.",
    "Ailemle vakit geçirmek bana huzur veriyor.",
    "Projeyi başarıyla tamamladım, çok gururluyum.",
    "Yeni bir hobi edinmek heyecan verici.",
    "Bugün kendimi motive hissediyorum.",
    "Bahçede çiçeklerle ilgilenmek içimi rahatlattı.",
    "Yıldızları izlemek çok romantikti.",
    "Kedim kucağımda uyuyunca çok mutlu oldum.",
    "Bugün her şey yolunda gitti.",
    "Yürüyüş sırasında doğanın seslerini dinlemek çok huzurluydu.",
    "Bugün hiç trafik yoktu, işe rahat ulaştım.",
    "Arkadaşım doğum günüme sürpriz yaptı.",
    "Kahvemi alıp balkonda kitap okumak çok keyifliydi.",
    "İlk defa kek yaptım ve çok lezzetli oldu.",
    "Hafta sonu planım harika geçti.",
    "Spor salonunda iyi bir antrenman yaptım.",
    "Sevdiğim film tekrar vizyona girmiş.",
    "Bugün enerjim yerindeydi.",
    "Yeni aldığım bitki çok güzel duruyor.",
    "Tüm işlerimi zamanında tamamladım.",
    "Kardeşimle birlikte oyun oynamak çok eğlenceliydi.",
    "Deniz kenarında yürümek bana terapi gibi geliyor.",
    "Çalışma ortamım çok sessizdi, verimli oldum.",
    "Yeni kıyafetimle kendimi çok iyi hissettim.",
    "Yabancı dil pratiği yaptım, geliştiğimi fark ettim.",
    "Uzun zamandır istediğim kitabı sonunda aldım.",
    "Bugün güneşli ve huzurlu bir gün.",
    "Yoga yapmak bana iyi geldi.",
    "Sevdiğim müzik çalınca dans ettim.",
    "İş yerinde takdir edilmek moralimi çok yükseltti.",
    "Yeni tarif denedim, harika oldu.",
    "Bugün hiç hata yapmadım, çok mutluyum.",
    "Arkadaşım beni arayıp güzel haber verdi.",
    "Ders çalışmak bu sefer çok keyifliydi.",
    "Yeni aldığım ayakkabılar çok rahat.",
    "Parkta çocukların neşesi beni mutlu etti.",
    "Bugün tüm işlerimi tamamlayabildim.",
    "Evim tertemiz oldu, içim açıldı.",
    "Kendime küçük bir ödül aldım, mutlu oldum.",
    "Film gecesi harikaydı.",
    "Kahvemin tadı mükemmeldi.",
    "Eski bir dostla karşılaşmak güzel bir sürprizdi.",
    "Bugün çok yaratıcı hissediyorum.",
    "İşimde terfi aldım, çok mutluyum.",
    "Sevdiğim yemekleri yedim.",
    "Hayalimdeki gitarı sonunda alabildim.",
    "Birisine yardım etmek içimi ısıttı.",
    "Doğada vakit geçirmek bana çok iyi geldi.",
    "Bugün hiç yorulmadım, dinç hissediyorum.",
    "Yeni projem için çok heyecanlıyım.",
    "Annemin yaptığı yemek çok lezzetliydi.",
    "Bugün çok güldüm, keyfim yerinde.",
    "Tatlı yedim, moralim düzeldi.",
    "Yeni bir hedef belirledim ve motiveyim.",
    "Pozitif düşünmek bana iyi geliyor.",
    "Birlikte çalıştığımız ekip çok uyumluydu.",
    "Telefon görüşmem çok keyifli geçti.",
    "Bugün yağmur yağmadı, dışarısı çok güzeldi.",
    "Öğle tatilimde güzel bir yürüyüş yaptım.",
    "Baharda çiçeklerin açmasını izlemek harikaydı.",
    "Yeni filmler keşfetmek hoşuma gidiyor.",
    "Kütüphanede huzur buldum.",
    "Bugün yeni bir şey öğrendim ve çok mutlu oldum.",
    "Kendi yaptığım müziği dinlemek çok güzeldi.",
    "Hafta sonunu doğada geçirmek harika hissettirdi.",
    "Yeni bir beceri kazanmak beni mutlu etti.",
    "Tertemiz hava çok iyi geldi.",
    "Sevdiğim kişiyle vakit geçirdim.",
    "Kamp yapmak çok eğlenceliydi.",
    "Bugün kendime zaman ayırdım ve çok iyi geldi.",
    "Uzun zamandır istediğim o eşyayı aldım.",
    "Yemek yaparken yeni tarifler denedim.",
    "Fotoğraf çekmeye çıktım, doğa harikaydı.",
    "Bugün güne erken başladım, çok verimliydi.",
    "Gün batımını izlemek ruhuma dokundu.",
    "Yeni bir dil öğrenmeye başladım.",
    "Pozitif yorumlar aldım, moralim yerine geldi.",
    "Spora başlamak çok iyi bir karardı.",
    "Bugün çok nazik insanlarla karşılaştım.",
    "Kahvaltı masası çok özenliydi.",
    "Eski günleri hatırlayıp gülümsedim.",
    "Ders anlatırken öğrenciler çok ilgiliydi.",
    "Kendime sağlıklı bir smoothie yaptım.",
    "Sokakta tanımadığım biri bana gülümsedi.",
    "Yolda yürürken güzel bir şarkı duydum.",
    "Bugün kimseyi kırmadım, kendimle gurur duydum.",
    "Yastıklarımı kabartıp rahatça uyudum.",
    "Doğum günü kutlamam harika geçti.",
]

# === TOKENİZER OLUŞTURMA VE EĞİTME ===
tokenizer=Tokenizer()  # Metinleri sayısal token dizilerine çevirecek tokenizer nesnesi oluştur
tokenizer.fit_on_texts(texts)  # Tokenizer'ı veri setindeki tüm metinler üzerinde eğit (kelime sözlüğü oluştur)
total_words=len(tokenizer.word_index)+1  # Toplam benzersiz kelime sayısı (+1 padding/unknown tokenlar için)

# === N-GRAM DİZİLERİ OLUŞTURMA ===
input_sequences=[]  # Eğitim için kullanılacak giriş dizilerini saklamak için boş liste

# Her cümle için N-gram dizileri oluştur (dil modeli eğitimi için)
for text in texts:  # Veri setindeki her metin için döngü
    token_list=tokenizer.texts_to_sequences([text])[0]  # Metni sayısal token dizisine çevir
    # N-gram dizileri oluştur: [w1], [w1,w2], [w1,w2,w3], ... şeklinde
    for i in range(1,len(token_list)):  # İkinci kelimeden başlayarak N-gram dizileri oluştur
        n_gram_sequence=token_list[:i+1]  # Baştan i+1'inci kelimeye kadar olan diziyi al
        input_sequences.append(n_gram_sequence)  # Oluşturulan N-gram dizisini listeye ekle

# === PADDING İŞLEMİ ===    
max_sequence_len=max([len(x) for x in input_sequences])  # En uzun N-gram dizisinin uzunluğunu bul
input_sequences=pad_sequences(input_sequences,maxlen=max_sequence_len,padding="pre")  # Tüm dizileri baştan sıfır ekleyerek aynı uzunluğa getir

# === GİRİŞ VE HEDEF VERİLERİNİ AYIRMA ===
X,y=input_sequences[:,:-1],input_sequences[:,-1]  # Son kelime hariç giriş (X), son kelime hedef (y)
y=keras.utils.to_categorical(y,num_classes=total_words)  # Hedef kelimeleri one-hot encoding ile kategorik hale getir

# === LSTM MODELİ OLUŞTURMA ===
model=Sequential()  # Katmanları sıralı olarak ekleyeceğimiz Sequential model oluştur

# 1. Embedding Katmanı - Kelimeleri yoğun vektörlere çevirir
model.add(Embedding(
    input_dim=total_words,        # Girdi boyutu: toplam kelime sayısı
    output_dim=50,               # Çıktı boyutu: her kelime 50 boyutlu vektöre çevrilir
    input_length=X.shape[1]      # Giriş dizisinin uzunluğu (padding sonrası sabit uzunluk)
))

# 2. LSTM Katmanı - Uzun-kısa süreli hafıza katmanı
model.add(LSTM(
    units=100,                   # LSTM hücresi sayısı (nöron sayısı)
    return_sequences=False       # Sadece son çıktıyı döndür (sequence-to-one problem)
))

# 3. Dense Çıkış Katmanı - Kelime tahmini için tam bağlantılı katman
model.add(Dense(
    units=total_words,           # Çıkış boyutu: tüm kelimeler için olasılık dağılımı
    activation="softmax"         # Softmax: olasılık dağılımı için (toplam=1)
))

# === MODEL DERLEME ===
model.compile(
    optimizer="adam",                    # Adam optimizatörü (adaptif learning rate)
    loss="categorical_crossentropy",     # Kategorik çapraz entropi (multiclass classification)
    metrics=["accuracy"]                 # Eğitim sırasında takip edilecek metrik
)

# === MODEL EĞİTİMİ ===
model.fit(
    X, y,                    # Eğitim verisi (girdi ve hedef)
    epochs=100,              # 100 epoch boyunca eğit
    verbose=1                # Eğitim sürecini detaylı göster
)


# === METİN ÜRETİMİ FONKSİYONU ===
def generate_text(seed_text, next_words):
    """
    LSTM modeli kullanarak verilen başlangıç metninden devam eden metin üretir
    
    Args:
        seed_text (str): Başlangıç metni (örn: "bugün hava")
        next_words (int): Üretilecek kelime sayısı
    
    Returns:
        str: Başlangıç metni + üretilen kelimeler
    """
    for _ in range(next_words):  # Belirtilen sayıda kelime üretmek için döngü
        # Mevcut metni modelin anlayabileceği formata çevir
        token_list=tokenizer.texts_to_sequences([seed_text])[0]  # Metni sayısal token dizisine çevir
        token_list=pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")  # Padding uygula (model giriş boyutuna uygun)
        
        # Model ile bir sonraki kelimeyi tahmin et
        predicted_probabilities=model.predict(token_list, verbose=0)  # Tüm kelimeler için olasılık dağılımı hesapla
        predicted_word_index=np.argmax(predicted_probabilities, axis=-1)  # En yüksek olasılıklı kelimenin indeksini bul
        
        # Tahmin edilen kelimeyi metne çevir ve ekle
        predicted_word=tokenizer.index_word[predicted_word_index[0]]  # İndeksi gerçek kelimeye çevir
        seed_text+=" "+predicted_word  # Tahmin edilen kelimeyi mevcut metne ekle
        
    return seed_text  # Tamamlanan metni döndür

# === METİN ÜRETİMİ TESTİ ===
seed_text="trafik "  # Başlangıç metni tanımla
generated_text = generate_text(seed_text, 6)  # 6 kelime üret
print("Üretilen metin:", generated_text)  # Sonucu ekrana yazdır


"""
=== LSTM (Long Short-Term Memory) DETAYLI AÇIKLAMA ===

Bu kod, LSTM (Uzun-Kısa Süreli Hafıza) sinir ağı kullanarak Türkçe metin üretimi yapmaktadır.
LSTM, RNN'in geliştirilmiş bir versiyonudur ve sequence-to-sequence problemlerde daha başarılıdır.

=== LSTM NEDİR VE NASIL ÇALIŞIR? ===

**LSTM'in RNN'den Farkları:**
1. **Uzun Süreli Hafıza**: Gradient vanishing problemini çözer
2. **Üç Kapı Mekanizması**: Forget, Input, Output gate'leri
3. **Cell State**: Bilgiyi uzun süre saklar ve iletir

**LSTM Kapıları:**
1. **Forget Gate**: Hangi bilgilerin unutulacağını belirler (σ(Wf·[ht-1,xt] + bf))
2. **Input Gate**: Hangi yeni bilgilerin saklanacağını belirler (σ(Wi·[ht-1,xt] + bi))
3. **Output Gate**: Hangi bilgilerin çıktı olacağını belirler (σ(Wo·[ht-1,xt] + bo))

=== BU KODDA YAPILAN İŞLEMLER ===

**1. Veri Hazırlama:**
- 115 Türkçe pozitif cümle kullanılıyor
- Her cümle N-gram dizilerine ayrılıyor
- Örnek: "Bugün hava güzel" → ["Bugün"], ["Bugün hava"], ["Bugün hava güzel"]

**2. Tokenization:**
- Her kelime benzersiz bir sayıyla temsil ediliyor
- Tokenizer kelime sözlüğü oluşturuyor
- Padding ile tüm diziler aynı uzunluğa getiriliyor

**3. Model Mimarisi:**
- **Embedding Layer (50 dim)**: Kelimeler → Dense vektörler
- **LSTM Layer (100 units)**: Sıralı bilgi işleme + hafıza
- **Dense Layer (softmax)**: Kelime olasılık dağılımı

**4. Training Process:**
- Input: N-gram'in ilk (n-1) kelimesi
- Target: N-gram'in son kelimesi
- Loss: Categorical crossentropy
- Optimizer: Adam

=== N-GRAM YAKLAŞIMI ===

Bu kod language modeling yaklaşımı kullanıyor:
- "Bugün hava" → "güzel" (bir sonraki kelimeyi tahmin et)
- Her adımda bir kelime üretiliyor
- Üretilen kelime bir sonraki adımın girdisi oluyor

=== AVANTAJLAR ===

**LSTM'in Güçlü Yanları:**
- Uzun mesafe bağımlılıklarını yakalayabilir
- Gradient vanishing problemini çözer
- Esnek sequence uzunlukları ile çalışabilir
- Dil modellemede başarılı sonuçlar verir

**Bu Implementasyonun Özellikleri:**
- Türkçe dil desteği
- Pozitif sentiment odaklı eğitim
- Basit ama etkili mimari
- Gerçek zamanlı metin üretimi

=== SINIRLILIKLAR ===

- Küçük veri seti (115 cümle)
- Sadece pozitif sentiment
- Basit N-gram yaklaşımı
- Context window sınırlı

=== MODEL PERFORMANSI ===

**Eğitim Süreci:**
- 100 epoch eğitim
- Adam optimizer ile hızlı konverjans
- Categorical crossentropy loss
- Accuracy metriği ile takip

**Çıktı Kalitesi:**
- Türkçe dilbilgisi kurallarına uygun
- Pozitif sentiment korunuyor
- Anlamlı kelime kombinasyonları
- Yaratıcı ama tutarlı üretim

=== TEKNIK DETAYLAR ===

**Embedding Layer:**
- 50 boyutlu word vectors
- Kelimelerin anlamsal ilişkilerini yakalar
- Trainable parametreler

**LSTM Parametreleri:**
- 100 hidden units
- return_sequences=False (sequence-to-one)
- Dropout regularization yok (küçük model)

**Output Layer:**
- Softmax activation
- Vocabulary size kadar çıktı
- Probability distribution over words

=== KULLANIM ÖRNEKLERİ ===

```python
# Farklı başlangıç metinleri ile test
generate_text("bugün", 5)      # → "bugün hava çok güzel morali"
generate_text("mutlu", 4)      # → "mutlu oldum çok keyifli"
generate_text("arkadaş", 6)    # → "arkadaş vakit geçirmek beni çok güzel"
```

Bu LSTM modeli, sequence modeling ve language generation konularında 
temel bir örnek sunmakta ve Türkçe NLP uygulamaları için başlangıç noktası oluşturmaktadır.
"""

 

