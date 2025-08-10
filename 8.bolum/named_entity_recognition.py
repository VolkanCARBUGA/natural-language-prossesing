# Named Entity Recognition (NER) - SpaCy ile Varlık Tanıma

import pandas as pd  # Veri manipülasyonu için pandas kütüphanesi
import spacy  # SpaCy doğal dil işleme kütüphanesi

# İngilizce SpaCy modelini yükleme
nlp=spacy.load("en_core_web_sm")  # Küçük İngilizce modeli yükleme

# Analiz edilecek metin - farklı türde varlıklar içeriyor
content="Alice works at Google and lives in london. She viisted the British Museum yesterday."

# Metni SpaCy ile işleme
doc=nlp(content)  # NLP pipeline'ından geçirme (tokenization, POS tagging, NER vb.)

# Bulunan her varlık için bilgileri yazdırma
for ent in doc.ents:
    print(ent.text,ent.start_char,ent.end_char,ent.label_)  # Varlık metni, başlangıç pozisyonu, bitiş pozisyonu, etiket

# Varlıkları liste olarak toplama
entities=[(ent.text,ent.label_,ent.lemma_)for ent in doc.ents]  # Her varlık için metin, etiket ve lemma bilgisi
print(entities)  # Varlık listesini yazdırma

# Varlıkları DataFrame'e dönüştürme
df=pd.DataFrame(entities,columns=["Entity","Label","Lemma"])  # Sütun isimleriyle DataFrame oluşturma
print(df)  # DataFrame'i yazdırma

# Sonuçları CSV dosyasına kaydetme
df.to_csv("8.bolum/entities.csv",index=False)  # Index olmadan CSV'ye kaydetme

# KONU AÇIKLAMASI:
# Named Entity Recognition (NER), metindeki kişi, yer, organizasyon, 
# tarih gibi önemli varlıkları tanımlama ve sınıflandırma işlemidir.
# SpaCy'nin önceden eğitilmiş modeli şu varlık türlerini tespit edebilir:
# - PERSON: Kişi adları (Alice)
# - ORG: Organizasyonlar (Google, British Museum)
# - GPE: Coğrafi politik varlıklar (London)
# - DATE: Tarihler (yesterday)
# Bu bilgiler metin analizi, bilgi çıkarımı ve arama sistemlerinde kritiktir.


