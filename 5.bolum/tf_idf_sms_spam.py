import pandas as pd  # Pandas kütüphanesini veri işleme için içe aktarıyoruz
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vektörizasyonu için scikit-learn'den gerekli sınıfı içe aktarıyoruz

df=pd.read_csv("5.bolum/spam.csv",encoding="latin-1")  # CSV dosyasını okuyoruz, encoding parametresi ile karakter kodlama sorunlarını çözüyoruz

# veri temizleme
df = df[['v1', 'v2']]  # Sadece gerekli sütunları (v1: etiket, v2: mesaj) alıyoruz
df.columns = ['label', 'message']  # Sütun isimlerini daha anlamlı hale getiriyoruz
df = df.dropna()  # Boş (null) değerleri içeren satırları kaldırıyoruz
df['message'] = df['message'].str.lower()  # Tüm metinleri küçük harfe çeviriyoruz (büyük-küçük harf tutarlılığı için)
df = df.drop_duplicates()  # Tekrarlayan satırları kaldırıyoruz

vectorizer=TfidfVectorizer()  # TF-IDF vektörizatör nesnesini oluşturuyoruz
X=vectorizer.fit_transform(df.message)  # Temizlenmiş mesajlara TF-IDF dönüşümü uyguluyoruz ve sparse matris elde ediyoruz

feature_names=vectorizer.get_feature_names_out()  # TF-IDF matrisindeki her sütunun hangi kelimeye karşılık geldiğini öğreniyoruz

tfidf_scores=X.mean(axis=0).A1  # Her kelimenin ortalama TF-IDF skorunu hesaplıyoruz (axis=0: sütunlar boyunca ortalama, A1: array'e çevirme)

df_tfidf=pd.DataFrame({  # TF-IDF sonuçlarını görselleştirmek için DataFrame oluşturuyoruz
    "word":feature_names,  # Kelimeler sütunu
    "tfidf":tfidf_scores   # Her kelimenin ortalama TF-IDF skoru sütunu
})

df_tfidf_sorted=df_tfidf.sort_values(by="tfidf",ascending=False)  # Kelimeleri TF-IDF skorlarına göre büyükten küçüğe sıralıyoruz
print(df_tfidf_sorted.head(10))  # En yüksek TF-IDF skoruna sahip ilk 10 kelimeyi ekrana yazdırıyoruz

"""
TF-IDF (Term Frequency - Inverse Document Frequency) AÇIKLAMA:

TF-IDF, doğal dil işlemede kelimelerin önemini ölçen bir tekniktir. İki bileşenden oluşur:

1. TF (Term Frequency - Terim Frekansı): 
   - Bir kelimenin bir belgede ne kadar sık geçtiğini ölçer
   - Formül: TF(t,d) = (Kelimenin belgede geçme sayısı) / (Belgedeki toplam kelime sayısı)

2. IDF (Inverse Document Frequency - Ters Belge Frekansı):
   - Bir kelimenin tüm koleksiyonda ne kadar nadir olduğunu ölçer
   - Formül: IDF(t) = log(Toplam belge sayısı / Kelimeyi içeren belge sayısı)

3. TF-IDF Skoru:
   - TF-IDF(t,d) = TF(t,d) × IDF(t)
   - Yüksek TF-IDF skoru = Kelime o belgede sık geçiyor AMA genel koleksiyonda nadir
   - Bu, belgeyi karakterize eden önemli kelimeleri bulmanızı sağlar

Bu kodda:
- SMS spam veri setini kullanarak her kelimenin ortalama TF-IDF skorunu hesaplıyoruz
- En yüksek skorlu kelimeler, spam mesajları ayırt etmede en etkili kelimelerdir
- Bu bilgi, spam filtreleme algoritmaları için kullanılabilir
"""



