# Lab 5 - LunarLanderContinuous-v3

Poniżej masz kolejność pracy, żeby zrobić to na wysoką liczbę punktów.

## Plan

1. Utwórz osobne środowisko `venv` dla tego zadania.
2. Zainstaluj zależności z `requirements.txt`.
3. Sprawdź, czy środowisko `LunarLanderContinuous-v3` uruchamia się bez błędów.
4. Zacznij od prostego baseline'u i porównaj co najmniej dwa algorytmy dla akcji ciągłych.
5. Trenuj model z sensownymi hiperparametrami i zapisuj logi do TensorBoard.
6. Oceniaj model na wielu epizodach, nie tylko na jednym.
7. Zrób wykresy i krótki opis wyników do raportu.

## Co warto zrobić, żeby zdobyć więcej punktów

- Dla `LunarLanderContinuous-v3` najlepiej zacząć od `SAC` albo `TD3`, bo środowisko ma akcje ciągłe.
- Porównaj wyniki z `PPO` jako baseline, ale traktuj go raczej jako punkt odniesienia niż główny algorytm.
- Użyj kilku seedów i pokaż średnią oraz odchylenie standardowe wyników.
- Zapisuj najlepszy model i końcowy model osobno.
- Dodaj ewaluację po treningu i krótką wizualizację działania agenta.

## Szybki start

```powershell
cd c:\Users\MateuszOmen\Desktop\studia\sem6\inteligencja_obliczeniowa_analiza_danych_cyfrowych\lab5\inteligencja_obliczeniowa_w_analizie_danych\lab5
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Kolejny krok

Po instalacji przygotujemy skrypt treningowy dla `LunarLanderContinuous-v3` i dopasujemy hiperparametry pod maksymalny wynik.