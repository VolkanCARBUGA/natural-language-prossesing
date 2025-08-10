# Morfolojik Analiz - SpaCy ile Kelime Düzeyinde Analiz

import spacy  # SpaCy doğal dil işleme kütüphanesi

# İngilizce SpaCy modelini yükleme
nlp = spacy.load("en_core_web_sm")  # Küçük İngilizce model (sm = small)

# Analiz edilecek kelime
word = "she"  # Test kelimesi olarak "she" zamir

# Kelimeyi SpaCy dokümani haline getirme
doc = nlp(word)  # SpaCy'nin NLP pipeline'ından geçirme

# Her token (kelime) için morfolojik özellikleri yazdırma
for token in doc:
    print(
        f"{token.text:10} {token.pos_:10} {token.dep_:10} {token.lemma_:10} "  # Metin, POS etiketi, bağımlılık ilişkisi, kök form
        f"{token.tag_:10} {token.shape_:10} {token.is_alpha:10} {token.dep:10} "  # Detaylı etiket, şekil, alfabetik mi, bağımlılık numarası
        f"{token.is_stop:10}"  # Stop word (gereksiz kelime) mi
    )

# KONU AÇIKLAMASI:
# Morfolojik analiz, kelimelerin yapısal özelliklerini inceler.
# SpaCy kullanarak bir kelime hakkında şu bilgileri elde ederiz:
# - text: Kelimenin kendisi
# - pos_: Kelime türü (PRON, NOUN, VERB vb.)
# - dep_: Cümle içindeki bağımlılık ilişkisi
# - lemma_: Kelimenin kök formu (lemma)
# - tag_: Daha detaylı gramer etiketi
# - shape_: Kelimenin şekli (büyük/küçük harf yapısı)
# - is_alpha: Sadece alfabetik karakterlerden oluşup oluşmadığı
# - is_stop: Stop word (gereksiz kelime) olup olmadığı
