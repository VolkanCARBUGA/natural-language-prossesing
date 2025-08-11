"""
9. BÖLÜM - METİN ÖZETLEME (TEXT SUMMARIZATION)

Bu dosya Transformer modelleri kullanarak otomatik metin özetleme yapar.

METİN ÖZETLEME NEDİR?
- Uzun metinleri kısa ve anlamlı özetlere çevirme süreci
- Ana fikirleri koruyarak bilgi yoğunluğunu azaltır
- Haber siteleri, araştırma makaleleri için kritik

ÖZETLEME TÜRLERİ:
1. EXTRACTIVE: Metnin içinden cümleler seçer
2. ABSTRACTIVE: Yeni cümleler ve ifadeler üretir (Bu projede kullanılan)

TRANSFORMER YAKLAŞIMI:
- Sequence-to-sequence modeller kullanır
- Attention mekanizması ile önemli kısımlara odaklanır
- Pre-trained modeller (BART, T5, Pegasus)

AVANTAJLARI:
✓ Özgün ve akıcı özetler
✓ Anlam bütünlüğü korur
✓ Çoklu dil desteği
✓ Farklı uzunluk seçenekleri

DEZAVANTAJLARI:
✗ Bazen gerçek dışı bilgi üretebilir
✗ Yüksek hesaplama maliyeti
✗ Uzun metinlerde performans düşüşü

KULLANIM ALANLARI:
- Haber özeti çıkarma
- Akademik makale özetleme
- Doküman yönetimi
- E-posta özetleme
- Sosyal medya içerik analizi
"""

# Hugging Face transformers kütüphanesinden summarization pipeline'ını içe aktar
from transformers import pipeline

# Özetleme için hazır pipeline oluştur (varsayılan model: sshleifer/distilbart-cnn-12-6)
summarizer = pipeline("summarization")

# Özetlenecek uzun metin (makine öğrenmesi hakkında)
text = """
Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use 
to progressively improve their performance on a specific task. Machine learning algorithms build a mathematical 
model of sample data, known as "training data", in order to make predictions or decisions without being explicitly 
programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, 
detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific 
instructions for performing the task. Machine learning is closely related to computational statistics, which focuses 
on making predictions using computers. The study of mathematical optimization delivers methods, theory and application 
domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory 
data analysis through unsupervised learning. In its application across business problems, machine learning is also referred 
to as predictive analytics.
"""

# Metni özetle: max_length=maksimum kelime sayısı, min_length=minimum kelime sayısı, do_sample=rastgele örnekleme kapalı
summary = summarizer(text, max_length=90, min_length=45,do_sample=False)
# Özet metnini yazdır (sonuç liste formatında gelir, ilk elemanın "summary_text" anahtarına erişim)
print(summary[0]["summary_text"])