# han/hafiza.py (GÜNCELLENMİŞ HALİ)
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Optional

class HafizaMotoru:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", # İstersen 7B yap
        embedder_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        load_in_4bit: bool = True  # <--- YENİ ÖZELLİK: Varsayılan 4-Bit
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 HAN başlatılıyor (Cihaz: {self.device.upper()})")
        
        # 4-BIT AYARLARI (RAM Tasarrufu)
        quantization_config = None
        if load_in_4bit and self.device == "cuda":
            print("💡 4-Bit Sıkıştırma Aktif (Düşük VRAM Modu)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

        print(f"📥 LLM yükleniyor: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config, # <--- EKLENDİ
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        print("📥 Embedding modeli yükleniyor...")
        self.embedder = SentenceTransformer(embedder_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        
        self.index = None
        self.stored_docs = []
        print("✅ HAN hazır!")

    # ... (Diğer fonksiyonlar: verileri_yukle, ara, soru_sor AYNI KALACAK) ...
    # Sadece yukarıdaki __init__ kısmını değiştirmen yeterli.
    
    # KODUN DEVAMINI AYNEN KORU (verileri_yukle, ara, soru_sor, kaydet, yukle)
    def verileri_yukle(self, metin_listesi: List[str], batch_size: int = 32):
        # ... (Aynı kod) ...
        if not metin_listesi:
            raise ValueError("Metin listesi boş olamaz")
        
        print(f"🔄 {len(metin_listesi)} belge işleniyor...")
        self.stored_docs = metin_listesi
        
        embeddings = self.embedder.encode(
            metin_listesi,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ {len(metin_listesi)} belge hafızaya eklendi")

    def ara(self, soru: str, k: int = 3, min_score: float = 2.0) -> List[str]:
        # ... (Aynı kod) ...
        if self.index is None or len(self.stored_docs) == 0:
            return []
        
        k = min(k, len(self.stored_docs))
        soru_vektoru = self.embedder.encode([soru], convert_to_numpy=True).astype('float32')
        mesafeler, indeksler = self.index.search(soru_vektoru, k)
        
        bulunanlar = []
        for mesafe, idx in zip(mesafeler[0], indeksler[0]):
            if idx < len(self.stored_docs): # min_score kontrolünü opsiyonel yapabilirsin
                bulunanlar.append(self.stored_docs[idx])
        
        return bulunanlar

    def soru_sor(self, soru: str, k: int = 3, max_tokens: int = 150, temperature: float = 0.1) -> str:
        # ... (Aynı kod) ...
        baglam_listesi = self.ara(soru, k=k)
        
        if not baglam_listesi:
            baglam = "İlgili bilgi bulunamadı."
        else:
            baglam = "\n---\n".join(baglam_listesi)
        
        messages = [
            {"role": "system", "content": "Sen yardımcı bir asistansın. Sadece verilen BAĞLAM bilgisini kullanarak cevap ver. Bilgi yoksa 'Bilmiyorum' de."},
            {"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def kaydet(self, dosya_adi: str):
        faiss.write_index(self.index, f"{dosya_adi}.index")
        np.save(f"{dosya_adi}_docs.npy", self.stored_docs)
        print(f"💾 Hafıza kaydedildi: {dosya_adi}")
    
    def yukle(self, dosya_adi: str):
        self.index = faiss.read_index(f"{dosya_adi}.index")
        self.stored_docs = np.load(f"{dosya_adi}_docs.npy", allow_pickle=True).tolist()
        print(f"📂 Hafıza yüklendi: {dosya_adi}")
