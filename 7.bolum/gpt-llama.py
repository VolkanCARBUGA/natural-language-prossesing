# Transformers kütüphanesinden GPT-2 modeli için gerekli sınıfları import et
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Transformers kütüphanesinden genel tokenizer ve causal LM modeli import et  
from transformers import AutoTokenizer, AutoModelForCausalLM

# GPT-2 modelinin HuggingFace Hub'daki adını tanımla
model_name="gpt2"
# Llama modelinin HuggingFace Hub'daki adını tanımla (7 milyar parametreli versiyon)
model_name_llama="huggyllama/llama-7b"

try:
    # === GPT-2 MODEL YÜKLEME ===
    # GPT-2 için özel tokenizer'ı indir ve yükle
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # GPT-2 dil modelini (Language Model Head ile) indir ve yükle
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # === LLAMA MODEL YÜKLEME ===
    # Llama modeli için genel tokenizer'ı indir ve yükle
    tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama)
    # Llama causal language modelini indir ve yükle
    model_llama = AutoModelForCausalLM.from_pretrained(model_name_llama)
    
    # === METİN TOKENİZASYONU ===
    # Test edilecek başlangıç metni tanımla
    input_text = "I go to school for"
    # GPT-2 tokenizer ile metni sayısal tokenlara çevir ve PyTorch tensor formatına dönüştür
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    # Llama tokenizer ile metni sayısal tokenlara çevir ve PyTorch tensor formatına dönüştür  
    input_llama = tokenizer_llama.encode(input_text, return_tensors="pt")
    
    # === METİN ÜRETİMİ ===
    # GPT-2 modeli ile metin üret
    # max_length=55: Maksimum 55 token uzunluğunda metin üret
    # do_sample=True: Rastgele örnekleme kullan (deterministik değil)
    # temperature=0.7: Yaratıcılık seviyesi (0=deterministik, 1=çok yaratıcı)
    output = model.generate(inputs, max_length=55, do_sample=True, temperature=0.7)
    # Llama modeli ile aynı parametrelerle metin üret
    output_llama = model_llama.generate(input_llama, max_length=55, do_sample=True, temperature=0.7)
    
    # === ÇIKTI DEKODLAMAsl ===
    # GPT-2 çıktısını tokenlerdan metne çevir
    # skip_special_tokens=True: Özel tokenları (padding, eos vs.) atla
    generate_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Llama çıktısını tokenlerdan metne çevir
    generate_text_llama = tokenizer_llama.decode(output_llama[0], skip_special_tokens=True)
    
    # === SONUÇLARI YAZDIRMA ===
    # GPT-2 modelinin ürettiği metni konsola yazdır
    print("GPT-2 Çıktısı:")
    print(generate_text)
    # Llama modelinin ürettiği metni konsola yazdır
    print("\nLlama Çıktısı:")
    print(generate_text_llama)
    
# Herhangi bir hata durumunda (model yüklenememe, memory yetersizliği vs.)
except Exception as e:
    print(f"Hata oluştu: {e}")
    print("Not: Llama modeli çok büyük olabilir ve yerel bilgisayarda çalışmayabilir.")


"""
=== GPT VE LLAMA MODELLERİ HAKKINDA DETAYLI AÇIKLAMA ===

Bu kod, iki farklı büyük dil modelini (Large Language Model - LLM) karşılaştırarak 
metin üretimi yapmaktadır: GPT-2 ve LLaMA.

=== GPT-2 (Generative Pre-trained Transformer 2) ===

**Ne Değildir:**
- GPT-2, OpenAI tarafından 2019 yılında geliştirilmiş transformer tabanlı bir dil modelidir
- 1.5 milyar parametreye sahip orta büyüklükte bir modeldir
- Çok büyük text veri setleri üzerinde önceden eğitilmiştir (pre-trained)

**Nasıl Çalışır:**
1. **Transformer Mimarisi**: Attention mekanizması kullanarak metindeki uzun mesafe bağımlılıklarını yakalar
2. **Autoregressive Üretim**: Bir sonraki kelimeyi, önceki tüm kelimeleri baz alarak tahmin eder
3. **Unsupervised Learning**: Etiketlenmemiş metin verilerinden dil yapısını öğrenir

**Özellikler:**
- Coherent (tutarlı) uzun metinler üretebilir
- Zero-shot task transfer: Özel eğitim olmadan farklı görevleri yapabilir
- İnsan yazısına benzer akıcı metinler üretir

=== LLaMA (Large Language Model Meta AI) ===

**Ne Değildir:**
- Meta (Facebook) tarafından 2023'te geliştirilmiş daha gelişmiş bir dil modelidir
- 7B, 13B, 30B, 65B parametreli farklı versiyonları vardır
- GPT-2'den çok daha büyük ve güçlüdür

**Özellikler:**
- **Efficiency**: Daha az parametre ile daha iyi performans
- **Better Training**: Daha kaliteli ve çeşitli veri setleri kullanılmış
- **Instruction Following**: Talimatlara daha iyi uyum sağlar
- **Multilingual**: Çoklu dil desteği daha gelişmiş

=== MODEL KARŞILAŞTIRMASI ===

**Boyut ve Performans:**
- GPT-2: 1.5B parametre, nispeten küçük ve hızlı
- LLaMA-7B: 7B parametre, daha büyük ve daha yetenekli

**Kullanım Alanları:**
- GPT-2: Prototipleme, eğitim, hafif uygulamalar
- LLaMA: Araştırma, gelişmiş NLP uygulamaları, profesyonel projeler

**Memory Gereksinimleri:**
- GPT-2: ~6GB RAM (CPU/GPU)
- LLaMA-7B: ~14GB+ RAM, tercihen GPU gerektirir

=== TEKNIK DETAYLAR ===

**Temperature Parametresi (0.7):**
- 0.0: Deterministik, her zaman aynı çıktı
- 0.5: Düşük yaratıcılık, tutarlı sonuçlar
- 0.7: Dengelenmiş yaratıcılık (bu kodda kullanılan)
- 1.0: Yüksek yaratıcılık, çeşitli ama bazen tutarsız sonuçlar

**Max Length (55):**
- Üretilecek maksimum token sayısı
- Token ≈ kelime veya kelime parçası
- Uzun metinler için daha yüksek değer gerekir

**Do Sample (True):**
- True: Probabilistic sampling (rastgele örnekleme)
- False: Greedy decoding (en yüksek olasılıklı token seçimi)

=== PRATIK KULLANIM ÖNERİLERİ ===

**GPT-2 İçin İdeal:**
- Eğitim amaçlı projeler
- Prototip geliştirme
- Küçük ölçekli uygulamalar
- Sınırlı donanım ortamları

**LLaMA İçin İdeal:**
- Profesyonel uygulamalar
- Araştırma projeleri
- Yüksek kalite gerektiren görevler
- Güçlü donanım mevcut olduğunda

Bu kod örneği, modern AI sistemlerinde kullanılan transformer tabanlı dil modellerinin
temel kullanımını göstermekte ve farklı model boyutlarının karşılaştırılmasına olanak sağlamaktadır.
"""