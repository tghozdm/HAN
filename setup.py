from setuptools import setup, find_packages

setup(
    name="han",
    version="0.1.0",
    packages=find_packages(),
    
    # LICENSE CONFIGURATION
    license="CC BY-NC-ND 4.0 (Non-Commercial, No-Derivatives)",
    
    author="Tolgahan Özdemir",

    
    description="Hybrid AI Navigator - Source Available (Non-Commercial)",
    
    # Bu kısım PyPI veya pip listelerinde görünür
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License", 
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    
    # Kullanıcılar kurarken bu uyarıyı görecek
    long_description="""
    # ⛔ LICENSE WARNING
    
    This software is protected under **CC BY-NC-ND 4.0**.
    
    1. ❌ **No Commercial Use:** Strictly prohibited.
    2. ❌ **No Modifications:** You cannot distribute changed versions.
    3. ✅ **Personal Use:** Allowed for research and education.
    
    Please contact the author for commercial licensing.
    """
)
