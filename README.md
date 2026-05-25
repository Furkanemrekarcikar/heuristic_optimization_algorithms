# Multi-Objective Diet Optimization Problem (MODP)

BLM20364E / BLM22332E — Heuristic Optimization Algorithms Term Project

Günlük öğün planını 405 hazır gıda arasından seçen, 3 hedefli çok amaçlı bir optimizasyon sistemi:
- **f1** — Kullanıcı tercihi (maksimize)
- **f2** — Maliyet (minimize)
- **f3** — CO₂ ayak izi (minimize)

Algoritmalar: **NSGA-II** ve **SPEA2**

---

## Gereksinimler

- Python 3.10+
- MySQL (MariaDB) — XAMPP veya standalone kurulum
- `diet` adlı veritabanı (proje SQL dosyasından yüklenmiş olmalı)

---

## Kurulum

### 1. Sanal ortam ve bağımlılıklar

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Veritabanı bağlantısı

Proje kök dizininde `.env.example` dosyasını kopyalayarak `.env` oluşturun:

```bash
cp .env.example .env
```

`.env` dosyasını kendi bilgilerinizle düzenleyin:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_user_name_here
DB_PASSWORD=your_password_here
DB_NAME=your_db_name_here
```

> `.env` dosyası `.gitignore` kapsamındadır — commit'e **girmez**.

XAMPP kullanıyorsanız Apache & MySQL servislerini başlatmayı unutmayın.

### 3. Kullanıcı ID'lerini doğrula

```bash
python scripts/list_users.py
```

Çıktıda görünen non-vegetarian ve vegetarian kullanıcı ID'lerini `src/config.py` içinde güncelleyin:

```python
USER1_ID = 1    # Non-vegetarian
USER2_ID = 2    # Vegetarian
```

---

## Proje Yapısı

```
heuristic_optimization_algorithms/
├── src/
│   ├── config.py                  # Tüm parametreler
│   ├── database/
│   │   ├── connection.py          # MySQL bağlantı sarmalayıcı
│   │   └── loader.py              # DataStore: besin, DRI, tercih verileri
│   ├── problem/
│   │   ├── chromosome.py          # Permütasyon temsili + greedy decoder
│   │   ├── objectives.py          # 3 hedef fonksiyonu
│   │   └── penalty.py             # DRI cezası + çeşitlilik cezası
│   ├── operators/
│   │   ├── crossover.py           # OX çaprazlama (projeksiyon düzeltmeli)
│   │   ├── mutation.py            # Swap mutasyon
│   │   └── selection.py           # Binary tournament seçim
│   ├── algorithms/
│   │   ├── nsga2.py               # NSGA-II tam implementasyon
│   │   └── spea2.py               # SPEA2 tam implementasyon
│   ├── metrics/
│   │   └── hypervolume.py         # Exact 3D hypervolume (slice method)
│   ├── experiment/
│   │   └── runner.py              # 3 deney tanımı + sonuç kaydetme
│   └── visualization/
│       └── plots.py               # Pareto, yakınsama, menü tablosu grafikleri
├── scripts/
│   ├── list_users.py              # DB kullanıcılarını listele
│   ├── test_db.py                 # DB bağlantı testi
│   ├── test_chromosome.py         # Kromozom & decoder testi
│   ├── test_penalty.py            # Ceza fonksiyonu testi
│   ├── test_operators.py          # Crossover / mutation / selection testleri
│   ├── test_nsga2.py              # NSGA-II smoke test
│   ├── test_spea2.py              # SPEA2 smoke test
│   └── run_experiments.py         # Deney koşturucu (ana giriş noktası)
├── results/                       # Otomatik oluşturulur, git'e eklenmez
├── requirements.txt
└── README.md
```

---

## Testleri Çalıştırma

Testleri sırasıyla çalıştırın — her biri bir öncekine bağımlıdır.

### DB bağlantı testi
```bash
python scripts/test_db.py
```
Beklenen çıktı: `Connection OK` ve birkaç örnek yemek adı.

### Kullanıcı listesi
```bash
python scripts/list_users.py
```
Beklenen çıktı: DB'deki tüm kullanıcılar tablo olarak.

### Kromozom & decoder testi
```bash
python scripts/test_chromosome.py
```
Beklenen çıktı: Geçerli permütasyon üretiliyor, decoder besin toplamlarını hesaplıyor.

### Ceza fonksiyonu testi
```bash
python scripts/test_penalty.py
```
Beklenen çıktı: DRI ve çeşitlilik cezaları hesaplanıyor, negatif değer yok.

### Operatör testleri (crossover / mutation / selection)
```bash
python scripts/test_operators.py
```
Beklenen çıktı:
```
Operator tests (500 trials each)
  Crossover : OK
  Mutation  : OK
  Selection : OK
All checks passed
```

### NSGA-II smoke test (küçük run)
```bash
python scripts/test_nsga2.py
```
Beklenen çıktı: Her iki kullanıcı için Pareto front oluşuyor, HV > 0.

### SPEA2 smoke test (küçük run)
```bash
python scripts/test_spea2.py
```
Beklenen çıktı: Archive dolu, non-dominated çözümler mevcut.

---

## Deneyleri Çalıştırma

### Hızlı test (pop=20, gen=10 — saniyeler içinde tamamlanır)
```bash
python scripts/run_experiments.py --quick
```

### Tek deney
```bash
python scripts/run_experiments.py --quick --exp 1   # Kullanıcı karşılaştırma
python scripts/run_experiments.py --quick --exp 2   # Algoritma karşılaştırma
python scripts/run_experiments.py --quick --exp 3   # Çeşitlilik etkisi
```

### Tam run (pop=100, gen=200 — birkaç dakika sürer)
```bash
python scripts/run_experiments.py
```

### Çıktılar (`results/` klasörü)

Her deney için aşağıdaki dosyalar üretilir:

| Dosya | İçerik |
|-------|--------|
| `pareto_front.csv` | Son Pareto front (ham hedefler) |
| `hv_history.csv` | Her nesil için hypervolume değeri |
| `best_menus.csv` | En iyi 3 çözümün yemek listesi |
| `summary.json` | Skaler metrikler (son HV, süre, front büyüklüğü) |
| `convergence.png` | HV yakınsama grafiği |
| `pareto_*.png` | Pareto front scatter grafikleri |
| `hv_bar.png` | Run'lar arası HV karşılaştırma çubuğu |
| `menu_*.png` | En iyi menü tablo görseli |

---

## Deney Açıklamaları

### Deney 1 — Kullanıcı Karşılaştırma
NSGA-II'yi User1 (omnivore) ve User2 (vejeteryan) için ayrı ayrı çalıştırır. İki kullanıcının Pareto frontlarını ve yakınsama hızlarını karşılaştırır.

### Deney 2 — Algoritma Karşılaştırma
Aynı kullanıcı ve parametrelerle NSGA-II ile SPEA2'yi karşılaştırır. Son hypervolume ve yakınsama hızı temel metriklerdir.

### Deney 3 — Çeşitlilik Cezasının Etkisi
NSGA-II'yi `alpha=0` (çeşitlilik kapalı) ve `alpha=0.5` (açık) ile çalıştırır. Çeşitlilik cezasının Pareto front kalitesine etkisini ölçer.

---

## Anahtar Parametreler (`src/config.py`)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `POP_SIZE` | 100 | Popülasyon büyüklüğü |
| `N_GEN` | 200 | Nesil sayısı |
| `P_CROSS` | 0.9 | Çaprazlama olasılığı |
| `LAMBDA_PENALTY` | 1.0 | Ceza ağırlığı λ |
| `ALPHA_DIVERSITY` | 0.5 | Çeşitlilik cezası ağırlığı α |
| `EPS_UPPER` | 1.15 | Üst DRI toleransı (%15 aşım izni) |
| `EPS_LOWER` | 0.90 | Alt DRI toleransı (%10 eksik izni) |
| `BREAKFAST_RATIO` | 0.35 | Kahvaltının günlük DRI payı |

---

## Olası Hatalar

**`mysql.connector.errors.ProgrammingError: Unknown database 'diet'`**
→ XAMPP'ta MySQL'e bağlanıp `diet` veritabanını oluşturun ve SQL dosyasını import edin.

**`Access denied for user 'root'@'localhost'`**
→ `src/config.py` içinde `DB_PASSWORD` değerini XAMPP MySQL şifrenizle güncelleyin.

**`ModuleNotFoundError: No module named 'mysql'`**
→ Sanal ortamın aktif olduğundan emin olun: `.venv\Scripts\activate`
