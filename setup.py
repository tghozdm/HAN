# setup.py (GÜNCELLENMİŞ HALİ)
from setuptools import setup, find_packages

setup(
    name="han",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "numpy>=1.24.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",  # <--- EKLENDİ (4-Bit için şart)
        "pypdf>=3.0.0",          # <--- EKLENDİ (PDF okuma için)
    ],
    author="Senin Adın",
    description="Hybrid AI Navigator - RAG system with FAISS + LLM",
    python_requires=">=3.8",
)
