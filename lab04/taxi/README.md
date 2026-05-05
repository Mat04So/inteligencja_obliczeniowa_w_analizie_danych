# Lab 04 — własne środowisko Gymnasium (`CustomGridTaxi-v0`)

Środowisko: siatka 5×5 z taksówką, odbiorem pasażera i dowozem do celu. Agent: **MaskablePPO** (Stable Baselines3 + sb3-contrib).

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

Zawsze uruchamiaj polecenia z **tego samego katalogu** (`lab04/taxi`), żeby importy (`custom_grid_taxi_env`) działały poprawnie.

### Losowa polityka (podgląd środowiska)

```bash
python main.py random
```

Opcjonalnie: `--seed 123`.

### Trening agenta

```bash
python train.py
```

Domyślnie ok. **300 000** kroków. Krótszy test:

```bash
python train.py --timesteps 80000 --seed 42
```

Modele zapisują się w folderze `models/`:

- `best_model.zip` — najlepszy wg ewaluacji w trakcie treningu
- `custom_grid_taxi_ppo.zip` — model po zakończeniu całego treningu

### Ewaluacja wytrenowanego modelu (jeden epizod w terminalu)

```bash
python main.py eval
```

Własna ścieżka do checkpointu:

```bash
python main.py eval --model models/best_model.zip --seed 0
```

## Uwagi

- Pierwsze uruchomienie `eval` wymaga wcześniejszego `python train.py` (albo gotowych plików w `models/`).
- Środowisko domyślnie używa obserwacji **dyskretnej** (indeks stanu 0–499). Tryb wektorowy: `gym.make("CustomGridTaxi-v0", observation_mode="vector")` — gorszy dla PPO; do eksperymentów w raporcie.
