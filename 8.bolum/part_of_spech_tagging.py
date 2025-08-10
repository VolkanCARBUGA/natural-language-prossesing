# Part-of-Speech (POS) Tagging - SpaCy ile Kelime Türü Etiketleme

import spacy  # SpaCy doğal dil işleme kütüphanesi

# İngilizce SpaCy modelini yükleme
nlp = spacy.load("en_core_web_sm")  # Küçük İngilizce modeli yükleme

# Test cümleleri
sentence1="I am a student"  # İlk test cümlesi
sentence2="She is a teacher"  # İkinci test cümlesi

# Cümleleri SpaCy ile işleme
doc1=nlp(sentence1)  # İlk cümleyi NLP pipeline'ından geçirme
doc2=nlp(sentence2)  # İkinci cümleyi NLP pipeline'ından geçirme

# İlk cümledeki her kelime için POS etiketini yazdırma
for token in doc1:
    print(token.text,token.pos_)  # Kelime ve POS etiketi (zamir, fiil, belirteç, isim)

# İkinci cümledeki her kelime için POS etiketini yazdırma
for token in doc2:
    print(token.text,token.pos_)  # Kelime ve POS etiketi (zamir, fiil, belirteç, isim)

# KONU AÇIKLAMASI:
# Part-of-Speech (POS) Tagging, cümledeki her kelimenin gramer açısından
# hangi türde olduğunu belirleme işlemidir. Temel POS etiketleri:
# - PRON: Zamir (I, she)
# - AUX: Yardımcı fiil (am, is)
# - DET: Belirteç (a)
# - NOUN: İsim (student, teacher)
# - VERB: Fiil
# - ADJ: Sıfat
# Bu etiketleme, cümle yapısını anlamak ve dilbilgisel analiz yapmak için temeldir.