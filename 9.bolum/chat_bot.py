"""
9. BÖLÜM - CHATBOT SİSTEMLERİ

Bu dosya OpenAI API kullanarak profesyonel bir chatbot sistemi oluşturur.

CHATBOT NEDİR?
- Kullanıcılarla doğal dilde konuşabilen yapay zeka sistemleri
- Müşteri hizmetleri, asistan uygulamaları, eğitim için kullanılır
- İnsan benzeri etkileşim sağlar

OPENAI API YAKLAŞIMI:
- En gelişmiş dil modelleri (GPT-3.5, GPT-4)
- Yüksek kaliteli, bağlamsal cevaplar
- Konuşma geçmişi yönetimi
- Sistem mesajları ile davranış kontrolü

AVANTAJLARI:
✓ Çok gelişmiş doğal dil anlayışı
✓ Kaliteli ve tutarlı cevaplar
✓ Çoklu dil desteği
✓ Sürekli güncellenen modeller

DEZAVANTAJLARI:
✗ İnternet bağlantısı gerekli
✗ API maliyeti var
✗ Rate limiting (istek sınırı)
✗ Veri gizliliği endişeleri
"""

# OpenAI kütüphanesini ve işletim sistemi modülünü içe aktarma
from openai import OpenAI
import os

# API anahtarınızı environment variable olarak ayarlayın: export OPENAI_API_KEY="your-key-here"
# Güvenlik için API anahtarını kodda sabit olarak yazmayın
# OpenAI API istemcisini oluşturma (API anahtarı ile)
#client = OpenAI(api_key="api key")

# Chatbot ile konuşma fonksiyonu - kullanıcı girdisini alır ve OpenAI API'sinden cevap döner
def chat_with_bot(prompt, history_list):
    # Konuşma geçmişini doğru formatta hazırla - OpenAI Chat API formatında mesaj listesi
    messages = []
    
    # Sistem mesajı ekle (chatbot'un davranışını tanımlar) - AI'nın nasıl davranacağını belirler
    messages.append({
        "role": "system", 
        "content": "Sen yardımsever bir asistansın. Türkçe sorulara Türkçe cevap veriyorsun."
    })
    
    # Konuşma geçmişini ekle - önceki mesajları chat context'ine dahil et
    for msg in history_list:
        messages.append(msg)
    
    # Mevcut kullanıcı mesajını ekle - şu anki soruyu listeye ekle
    messages.append({"role": "user", "content": prompt})
    
    # API çağrısını try-catch bloğu içinde yap (hata yakalama için)
    try:
        # OpenAI Chat Completions API'sine istek gönder
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Kullanılacak AI modeli
            messages=messages,      # Mesaj geçmişi
            max_tokens=500,         # Maksimum cevap uzunluğu
            temperature=0.7         # Yaratıcılık seviyesi (0-1 arası)
        )
        # Cevabı temizleyip döndür
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Hata durumunda kullanıcıya bilgi ver
        return f"Hata oluştu: {str(e)}"

# Ana program çalıştığında burası başlar
if __name__=="__main__":
    # Konuşma geçmişini saklamak için boş liste oluştur
    history_list=[]
    # Kullanıcıya hoş geldin mesajı göster
    print("Chatbot başlatıldı! Çıkmak için 'exit', 'q' veya 'bye' yazın.")
    print("-" * 50)
    
    # Sonsuz döngü - kullanıcı çıkana kadar devam et
    while True:
        # Kullanıcıdan girdi al
        user_input=input("Mesajınız: ")
        # Çıkış komutlarını kontrol et
        if user_input.lower() in ["exit","q","bye","çıkış"]:
            print("Bot: Hoşça kalın!")
            break  # Döngüden çık
        
        # Chatbot'tan cevap al
        response=chat_with_bot(user_input, history_list)
        
        # Konuşma geçmişine kullanıcı mesajını ve bot cevabını ekle
        history_list.append({"role":"user","content":user_input})
        history_list.append({"role":"assistant","content":response})
        
        # Geçmişi maksimum 10 mesajla sınırla (performans ve token limiti için)
        if len(history_list) > 10:
            history_list = history_list[-10:]  # Son 10 mesajı tut
        
        # Bot cevabını kullanıcıya göster
        print(f"Bot: {response}")
        print("-" * 30)  # Ayırıcı çizgi