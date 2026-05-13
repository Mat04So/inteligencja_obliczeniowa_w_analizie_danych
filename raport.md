# Raport: własne środowisko Taxi i trening MaskablePPO

**Autorzy:** Antoni Pater, Mateusz Sobiech

## Cel ćwiczenia

Celem projektu było przygotowanie własnego środowiska typu Taxi w bibliotece Gymnasium oraz wytrenowanie agenta uczącego się dowożenia pasażera do celu. Agent korzysta z algorytmu MaskablePPO, czyli wariantu PPO obsługującego maskowanie niedozwolonych akcji.

Końcowy wariant środowiska działa na siatce `10x10`, zawiera przeszkody, losowe rozmieszczenie punktów `R/G/Y/B` oraz losowo wybieranego pasażera i cel. Dzięki temu agent nie uczy się jednej stałej trasy, tylko musi reagować na aktualny układ zadania.

Raport opisuje finalną wersję środowiska, sposób uczenia agenta, interpretację krzywej uczenia oraz przykładowe zrzuty ekranu pokazujące różne rozmieszczenia pól na planszy.

## Środowisko

Środowisko znajduje się w pliku `lab04/taxi/custom_grid_taxi_env.py`, a finalny notebook do uruchamiania treningu i rysowania krzywej uczenia to `lab04/taxi/lab04_taxi_final.ipynb`.

Najważniejsze cechy środowiska:

- plansza ma rozmiar `10x10`,
- część pól jest zablokowana jako ściany/przeszkody,
- taxi nie może wjechać w przeszkodę,
- punkty `R/G/Y/B` są losowane na wolnych polach przy każdym resecie,
- pasażer startuje w jednym z punktów `R/G/Y/B`,
- cel jest losowany jako inny punkt niż punkt startowy pasażera,
- taxi startuje z losowego wolnego pola,
- epizod kończy się sukcesem po poprawnym dowiezieniu pasażera do celu.

## Losowe rozmieszczenie punktów

W klasycznym Taxi punkty odbioru i dostawy są zwykle stałe, np. w rogach mapy. W tym projekcie zostało to zmienione. Przy każdym wywołaniu `reset()` środowisko losuje cztery różne wolne pola i przypisuje im etykiety `R`, `G`, `Y`, `B`.

Takie podejście sprawia, że:

- agent nie może zapamiętać jednej konkretnej trasy,
- pasażer i cel mogą znajdować się w różnych miejscach w kolejnych epizodach,
- środowisko jest trudniejsze i bardziej uogólnione,
- obserwacja musi zawierać informację o aktualnym źródle, celu i pozycji taxi.

Przeszkody pozostają stałe, ale punkty `R/G/Y/B`, pasażer, cel oraz pozycja taxi są losowe. Dzięki temu na tej samej mapie powstaje wiele różnych wariantów zadania.

## Obserwacja i akcje

Agent otrzymuje obserwację wektorową typu `Box(10,)`. Wektor zawiera znormalizowane informacje o:

- pozycji taxi,
- stanie pasażera,
- pozycji źródła pasażera,
- pozycji celu,
- aktualnym celu nawigacyjnym,
- odległości do aktualnego celu liczonej najkrótszą ścieżką BFS.

Dostępne akcje:

| Numer | Akcja |
|---:|---|
| 0 | ruch w dół |
| 1 | ruch w górę |
| 2 | ruch w prawo |
| 3 | ruch w lewo |
| 4 | pickup |
| 5 | dropoff |

Środowisko zwraca maskę akcji. Maska blokuje m.in. ruch w ścianę, pickup w złym miejscu oraz dropoff poza właściwym celem.

## Nagrody

Nagrody zostały dobrane tak, aby agent otrzymywał czytelny sygnał uczenia:

| Zdarzenie | Nagroda |
|---|---:|
| każdy krok | `-0.2` |
| poprawny pickup | `+10` |
| poprawny dropoff | `+100` |
| błędny pickup/dropoff | `-10` |
| zbliżenie do aktualnego celu | `+2.0 * (d_stare - d_nowe)` |

Odległość `d` jest liczona algorytmem BFS, a nie prostym Manhattanem. Jest to ważne, ponieważ na planszy występują ściany. Dystans Manhattan mógłby sugerować ruch w stronę przeszkody, natomiast BFS uwzględnia faktycznie dostępne przejścia.

## Algorytm

Do treningu użyto `MaskablePPO` z pakietu `sb3_contrib`. Jest to PPO rozszerzone o obsługę masek akcji. Dzięki temu agent nie losuje akcji niemożliwych do wykonania, np. wjazdu w ścianę.

Najważniejsze hiperparametry:

| Parametr | Wartość |
|---|---:|
| liczba kroków treningu | `3_000_000` |
| liczba środowisk równoległych | `8` |
| `learning_rate` | liniowy harmonogram od `2.5e-4` |
| `n_steps` | `512` |
| `batch_size` | `256` |
| `n_epochs` | `8` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.2` |
| `ent_coef` | `0.01` |
| `vf_coef` | `0.5` |
| `target_kl` | `0.03` |
| architektura sieci | `pi=[128, 128]`, `vf=[128, 128]` |

Model zapisuje się do:

```text
lab04/taxi/models/v1_10x10_walls_random_locs_vector/best_model.zip
```

## Krzywa uczenia

Krzywa uczenia jest najważniejszym wynikiem eksperymentu, ponieważ pokazuje, czy agent faktycznie nauczył się realizować zadanie, a nie tylko wykonywać pojedyncze poprawne akcje. Wykres został wygenerowany w notebooku `lab04/taxi/lab04_taxi_final.ipynb` na podstawie okresowych ewaluacji modelu.

![Krzywa uczenia agenta](krzywa-uczenia.png)

Dane do wykresu są zapisywane w pliku:

```text
lab04/taxi/models/logs_10x10_walls_random_locs_vector/evaluations.npz
```

Wykres składa się z trzech części, które razem pokazują jakość wyuczonej polityki:

- średnia nagroda epizodu,
- średnia długość epizodu,
- skuteczność dostaw w procentach.

Na początku treningu skuteczność jest niska, ponieważ agent dopiero poznaje znaczenie akcji, punktów `R/G/Y/B` i masek akcji. W kolejnych etapach rośnie średnia nagroda, maleje średnia długość epizodu, a odsetek udanych dostaw przekracza cel `90%`. Oznacza to, że agent nauczył się nie tylko dowozić pasażera, ale robić to coraz krótszą trasą.

Najważniejszy jest trzeci wykres, ponieważ pokazuje, jaki odsetek epizodów zakończył się poprawnym dowiezieniem pasażera do celu. Dodatkowo długość epizodu pokazuje, czy agent uczy się dostarczać pasażera szybciej, a średnia nagroda pokazuje ogólny postęp polityki.

## Uruchomienie

Trening i krzywa uczenia są uruchamiane z notebooka:

```text
lab04/taxi/lab04_taxi_final.ipynb
```

Notebook należy uruchomić od góry. Najważniejsze sekcje:

- `3. Trening MaskablePPO` uruchamia trening,
- `4. Krzywa uczenia` rysuje wykres,
- `5. Przykładowa ewaluacja agenta` uruchamia jeden epizod gry.

W ostatniej sekcji można zmienić tryb:

```python
RUN_MODE = "ansi"
```

na:

```python
RUN_MODE = "human"
```

Wtedy uruchamia się okno `pygame` z graficznym podglądem gry.

## Zrzuty ekranu gry

Poniższe obrazy pokazują graficzny podgląd środowiska. Każdy reset może wygenerować inne rozmieszczenie punktów `R/G/Y/B`, inną pozycję taxi oraz inny wybór pasażera i celu. Dzięki temu plansza nie sprowadza się do jednej zapamiętanej trasy.

### Przykład 1: losowe rozmieszczenie punktów

![Pierwszy układ gry](first-img.png)

Opis: punkty `R/G/Y/B` znajdują się w losowych miejscach na wolnych polach mapy. Taxi startuje z losowego pola, a pasażer i cel są wybrane z aktualnych punktów.

### Przykład 2: inny układ po zmianie seed/reset

![Drugi układ gry](drugi-obraz.png)

Opis: po zmianie seeda lub kolejnym resecie punkty `R/G/Y/B` są rozmieszczone inaczej. To pokazuje, że środowisko nie jest jedną zapamiętaną trasą.

### Znaczenie losowych układów

Losowe rozmieszczenie punktów jest ważne dla oceny jakości agenta. Gdyby punkty były zawsze w tych samych miejscach, agent mógłby nauczyć się prostego schematu poruszania po mapie. W finalnej wersji musi odczytać aktualną obserwację, odnaleźć pasażera, wykonać `pickup`, a następnie dowieźć go do aktualnie wskazanego celu.

## Wnioski

Zastosowanie masek akcji znacząco upraszcza problem, ponieważ agent nie musi eksplorować ruchów niemożliwych, takich jak wejście w ścianę. Dodatkowo shaping oparty o BFS daje poprawny sygnał kierunku nawet na mapie z przeszkodami.

Losowe rozmieszczenie punktów `R/G/Y/B` zwiększa trudność zadania, ale poprawia ogólność rozwiązania. Agent nie uczy się jednej trasy, tylko musi wykorzystywać obserwację zawierającą aktualne pozycje źródła, celu i taxi.

Końcowy notebook zawiera pełny proces: definicję konfiguracji, trening MaskablePPO, zapis modelu, krzywą uczenia oraz uruchomienie przykładowego epizodu gry.
