# Lab 04 - własne środowisko Gymnasium

Dwa środowiska na siatce 5×5, agent: **MaskablePPO** (Stable Baselines3 + sb3-contrib).

| Wersja | ID                    | Opis                                                        |
| ------ | --------------------- | ----------------------------------------------------------- |
| **v1** | `CustomGridTaxi-v0`   | 1 pasażer, obserwacja dyskretna (0-499)                     |
| **v2** | `CustomGridTaxi2P-v0` | 2 pasażerów, taxi wiezie jednego naraz, obserwacja Box(12,) |

## Wymagania

- Python 3.10+ (testowane z 3.12)
- [PyTorch](https://pytorch.org/) (instaluje się razem z zależnościami poniżej)

## Instalacja

Z katalogu `lab04/taxi`:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie

Zawsze uruchamiaj z **katalogu `lab04/taxi`**, żeby importy działały poprawnie.

### Losowa polityka (podgląd środowiska)

```bash
python main.py random               # v1, okno graficzne
python main.py random --env v2      # v2 (2 pasażerów), okno graficzne
```

Domyślnie otwiera się **okno graficzne (pygame)** z taksówką, pasażerami, celami i HUD-em.
Tryb tekstowy lub bez ekranu:

```bash
python main.py random --render ansi
python main.py random --env v2 --seed 7 --fps 6
```

### Trening agenta

```bash
python train.py                               # v1, 300 000 kroków (~70 s na CPU)
python train.py --env v2                      # v2, 3 000 000 kroków (domyślnie)
python train.py --env v2 --timesteps 4000000 # dłuższy trening v2
python train.py --env v2 --n-envs 8          # szybsze uczenie (równolegle)
```

Modele zapisują się w `models/`:

| Plik                          | Opis                    |
| ----------------------------- | ----------------------- |
| `best_model.zip`              | najlepszy checkpoint v1 |
| `custom_grid_taxi_ppo.zip`    | finalny model v1        |
| `best_model_v2.zip`           | najlepszy checkpoint v2 |
| `custom_grid_taxi_ppo_v2.zip` | finalny model v2        |

### Ewaluacja wytrenowanego modelu (okno graficzne)

```bash
python main.py eval                          # v1, best_model.zip
python main.py eval --env v2                 # v2, best_model_v2.zip
python main.py eval --env v2 --fps 6 --seed 5
python main.py eval --render ansi            # bez okna, ASCII
```

## Uwagi

- Pierwsze uruchomienie `eval` wymaga wcześniejszego `python train.py` (albo gotowych plików w `models/`).
- v1 używa obserwacji **dyskretnej** (indeks 0-499). Tryb wektorowy dostępny przez `observation_mode="vector"` - do eksperymentów w raporcie.
- v2 używa wektorowej obserwacji **Box(12,)**: pozycja taxi, status i współrzędne src/dst każdego pasażera.
- Nagrody v2: **+20** za pierwszą dostawę, **+30** za drugą (bonus za ukończenie misji), **−1** za krok.
- Tryby renderowania: `ansi`, `human` (pygame) oraz `rgb_array` (klatka jako `np.ndarray (H,W,3)`).
- Trening nie otwiera okna graficznego - szybciej, działa też na SSH. Wizualizację uruchomisz przez `main.py eval`.

---

## 8. Wizualizacja graficzna - uruchamianie przez terminal

> Poniższe komendy uruchamiamy **w terminalu**, nie w notebooku - pygame wymaga własnego okna.

```bash
# przejdź do katalogu projektu i aktywuj środowisko
cd lab04/taxi
source .venv/bin/activate

# --- v1 ---
python main.py random                  # losowa polityka, okno pygame
python main.py eval                    # wytrenowany agent v1
python main.py eval --fps 6            # szybciej
python main.py eval --render ansi      # tylko tekst (bez okna)

# --- v2 (2 pasażerowie) ---
python main.py random --env v2
python main.py eval   --env v2
python main.py eval   --env v2 --fps 5 --seed 3

# --- trening od zera ---
python train.py                        # v1, 300k kroków (~70s)
python train.py --env v2               # v2, 3M kroków (domyślnie)
```

### Co widać w oknie pygame

| Element             | v1                   | v2                                 |
| ------------------- | -------------------- | ---------------------------------- |
| Siatka 5×5 + ściany | ✓                    | ✓                                  |
| Stacje R/G/Y/B      | szare etykiety       | szare etykiety                     |
| Pasażer czekający   | niebieska postać     | P1=niebieski, P2=pomarańczowy      |
| Cel dostawy         | fioletowa ramka      | niebieska/pomarańczowa ramka P1/P2 |
| Taxi puste          | żółte                | żółte                              |
| Taxi z pasażerem    | zielone              | niebieski (P1) / pomarańczowy (P2) |
| HUD (dół)           | krok, nagroda, akcja | trasy P1/P2, krok, nagroda, akcja  |
