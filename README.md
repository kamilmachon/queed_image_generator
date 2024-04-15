# Queed NFT generator

## 1. Setup
### 1.1. Instalacja Pythona 3.10.8

Zainstaluj Python 3.10.8 i dodaj go do zmiennej PATH:
- [Python Release Python 3.10.8 | Python.org](https://www.python.org/downloads/release/python-3108/)

### 1.2. Konfiguracja wirtualnego środowiska

**Linux & MacOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 1.3. Instalowanie zależności

#### Dla systemów z GPU (min. 10 GB VRAM):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

#### Dla systemów CPU:
```bash
pip install -r requirements.txt
```

### 1.4. Uzupełnij plik konfiguracyjny do fetch ai oraz S3 

### 1.5. Uruchom agenta na Cascade (CPU) lub SDXL (GPU)
```bash
python image_generating_agent.py --model cpu
``` 
```bash
python image_generating_agent.py --model gpu
```

## 2. Użytkowanie modeli

### [SDXL 1.0 (Stable Diffusion Extra Large)](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)

#### Generowanie obrazów
- **Prompt**: Użycie słów kluczowych "pixelart" i "vectorized" dla generowania obrazów pomaga nadać charakter tokena NFT.
- **Rozdzielczość**: Model obsługuje wyższe rozdzielczości dla lepszej jakości i szczegółowości, minimalnie 512x512, zalecane 1024x1024px.
- **CFG Scale**: Kontroluje zgodność obrazu z promptem; wyższe wartości zwiększają zgodność, ale mogą wpływać na sztuczność detali.

#### Modyfikowanie obrazów z użyciem refinera
- **Strength (0-1)**: Określa stopień zgodności refiner z oryginalnym obrazem; wyższe wartości oznaczają większą zgodność.
- **CFG Scale**: Jak w przypadku generowania z promptu.

### [Cascade Lite](https://huggingface.co/stabilityai/stable-cascade?text=pixelart+vectorized+dog+riding+motorcycle+with+crazy+hat)
Model na zbudowany architekturze Würstchen - pracuje w znacząco mniejszej przestrzeni 24x24, co pozwala na szybkie użytkowanie go niskim kosztem, tym samym pracy na CPU. Generuje znacząco gorsze wyniki w porównaniu do SDXL i jest rekomendowany do użycia tylko w celach testowych.

### Test wydajności

- SDXL 1.0 1024x1024 na GPU (RTX 4080): ~20s
- SDXL 1.0 1024x1024 na CPU (i7-13700K + ~40GB RAM): ~18min
- Cascade CPU 512x512 (AMD Ryzen 5 5600h): ~4 min
