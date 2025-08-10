# Metin Sınıflandırma - Spam Tespiti için Decision Tree Classifier

import pandas as pd  # Veri manipülasyonu için pandas kütüphanesi
import nltk  # Doğal dil işleme kütüphanesi
import re  # Düzenli ifadeler için regex kütüphanesi
from nltk.corpus import stopwords  # Gereksiz kelimeleri filtrelemek için
from nltk.stem import WordNetLemmatizer  # Kelimeleri kök formlarına dönüştürmek için
from sklearn.model_selection import train_test_split  # Veriyi eğitim-test olarak bölmek için
from sklearn.feature_extraction.text import CountVectorizer  # Bag-of-Words vektörizasyonu için
from sklearn.tree import DecisionTreeClassifier  # Karar ağacı sınıflandırıcısı
from sklearn.metrics import confusion_matrix  # Model performans değerlendirme

# Spam veri setini yükleme
data = pd.read_csv("8.bolum/metin_siniflandirma_spam_veri_seti.csv", encoding="latin-1")

# Gereksiz sütunları silme
data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)  # Boş sütunları kaldırma
data.columns=["label","text"]  # Sütun isimlerini düzenleme
print(data.isnull().sum())  # Eksik veri kontrolü

# Gerekli NLTK veri setlerini indirme
nltk.download('stopwords')  # İngilizce stop words
nltk.download('wordnet')  # WordNet lemmatization için
nltk.download('omw-1.4')  # Çok dilli WordNet desteği

# Metin verilerini alma ve ön işleme hazırlığı
text=list(data.text)  # Tüm metinleri listeye çevirme
lemmatizer=WordNetLemmatizer()  # Lemmatizer nesnesi oluşturma
corpus=[]  # İşlenmiş metinleri tutacak liste

# Her metin için ön işleme uygulama
for i in range(len(text)):
    r=re.sub('[^a-zA-Z]',' ',text[i])  # Sadece harfleri tutma, diğer karakterleri boşlukla değiştirme
    r=r.lower()  # Küçük harfe çevirme
    r=r.split()  # Kelimelere ayırma
    r=[word for word in r if word not in set(stopwords.words('english'))]  # Stop words'leri filtreleme
    r=[lemmatizer.lemmatize(word) for word in r]  # Kelimeleri kök formlarına dönüştürme
    r=' '.join(r)  # İşlenmiş kelimeleri tekrar birleştirme
    corpus.append(r)  # İşlenmiş metni korpusa ekleme

data["text2"]=corpus  # İşlenmiş metinleri yeni sütun olarak ekleme

# Özellik ve hedef değişkenleri ayırma
X=data["text2"]  # Bağımsız değişken (işlenmiş metin)
y=data["label"]  # Bağımlı değişken (spam/ham etiketi)

# Veriyi eğitim ve test setlerine bölme
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)  # %80 eğitim, %20 test

# Bag-of-Words vektörizasyonu
count_vectorizer=CountVectorizer()  # CountVectorizer nesnesi oluşturma
X_train_count_vectorizer=count_vectorizer.fit_transform(X_train)  # Eğitim verilerine fit edip transform etme

# Decision Tree sınıflandırıcısını eğitme
decision_tree_classifier=DecisionTreeClassifier()  # Karar ağacı modeli oluşturma
decision_tree_classifier.fit(X_train_count_vectorizer,y_train)  # Modeli eğitme

# Test verilerini vektörize etme
X_test_count_vectorizer=count_vectorizer.transform(X_test)  # Test verilerini transform etme (sadece transform!)

# Tahmin yapma ve performans değerlendirme
prediction=decision_tree_classifier.predict(X_test_count_vectorizer)  # Test seti üzerinde tahmin yapma
c_matrix=confusion_matrix(y_test,prediction)  # Karışıklık matrisi hesaplama
accuracy=100*(sum(sum(c_matrix))-c_matrix[1,0]-c_matrix[0,1])/sum(sum(c_matrix))  # Doğruluk oranı hesaplama
print(f"Accuracy: {accuracy:.2f}%")  # Doğruluk oranını yazdırma

# KONU AÇIKLAMASI:
# Metin sınıflandırma, metinleri önceden tanımlanmış kategorilere ayırma işlemidir.
# Bu örnekte SMS mesajları spam/ham (gereksiz/gerekli) olarak sınıflandırılmaktadır.
# İşlem adımları:
# 1. Metin ön işleme: Temizleme, küçük harfe çevirme, stop words kaldırma, lemmatization
# 2. Vektörizasyon: Bag-of-Words ile metinleri sayısal vektörlere dönüştürme
# 3. Model eğitimi: Decision Tree ile sınıflandırıcı eğitme
# 4. Değerlendirme: Test seti üzerinde performans ölçme
# Bu yöntem email filtreleme, haber kategorilendirme gibi birçok alanda kullanılır.






 


    
    
    




