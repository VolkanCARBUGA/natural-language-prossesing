# Gelişmiş Kelime Anlam Belirsizliği Çözümü - PyWSD ile Farklı Lesk Algoritmaları

import nltk  # Doğal dil işleme kütüphanesi
nltk.download('averaged_perceptron_tagger_eng')  # İngilizce POS (Part-of-Speech) tagging modeli
from pywsd.lesk import simple_lesk,adapted_lesk,cosine_lesk  # PyWSD kütüphanesinden farklı Lesk algoritmaları

# Test cümleleri - aynı kelime farklı bağlamlarda
sentences=["I go to the bank to deposit money","The river bank is flooded after the heavy rain"]
word="bank"  # Analiz edilecek kelime

# Her cümle için farklı Lesk algoritmalarını test etme
for s in sentences:
    print(f"Cümle: {s}")  # Analiz edilen cümleyi yazdırma
    
    # Üç farklı Lesk algoritması ile kelime anlamını belirleme
    sense1=simple_lesk(s,word)  # Basit Lesk algoritması - temel örtüşme hesabı
    sense2=adapted_lesk(s,word)  # Uyarlanmış Lesk - daha gelişmiş örtüşme hesabı
    sense3=cosine_lesk(s,word)  # Kosinüs Lesk - vektör benzerliği kullanarak
    
    # Her algoritmanın bulduğu anlamı yazdırma
    print(f"Simple Lesk: {sense1.definition()}")  # Basit Lesk sonucu
    print(f"Adaptive Lesk: {sense2.definition()}")  # Uyarlanmış Lesk sonucu
    print(f"Cosine Lesk: {sense3.definition()}")  # Kosinüs Lesk sonucu
    print("-"*50)  # Ayırıcı çizgi

# KONU AÇIKLAMASI:
# Bu kod, kelime anlam belirsizliği için üç farklı Lesk algoritmasını karşılaştırır:
# 1. Simple Lesk: Temel örtüşme sayma yöntemi
# 2. Adapted Lesk: İlişkili kelimeleri de dahil eden geliştirilmiş versyon
# 3. Cosine Lesk: Vektör uzayında kosinüs benzerliği kullanan modern yaklaşım
# Her algoritma farklı güçlü yanları olan farklı yaklaşımlar sunar ve
# farklı durumlarda daha iyi performans gösterebilir.