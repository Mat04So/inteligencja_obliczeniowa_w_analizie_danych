# Projekt 1 EasyAI - TicTacDoh (wariant probabilistyczny)

## Zakres wykonania (wymagania obowiazkowe do 6 pkt)

1. Wykonano wariant probabilistyczny gry Tic-tac-doh:
	- plik: easyAI-main/lab1/probabilistic_tic_tac_doh.py
	- z prawdopodobienstwem 20% ruch sie nie udaje i plansza pozostaje bez zmian.

2. Wykonano porownanie wymaganych algorytmow:
	- Negamax z odcieciem alpha-beta (negamax_ab)
	- Negamax bez odciecia alpha-beta (negamax_no_ab)
	- dwie glebokosci przeszukiwania: 2 oraz 4
	- wariant deterministyczny i probabilistyczny gry.

3. Wykonano pomiar sredniego czasu wyboru ruchu dla kazdego wariantu AI.


## Kod eksperymentu

- Skrypt porownan: easyAI-main/lab1/compare_algorithms_6pts.py

Uruchomienie uzyte do pomiarow:

python easyAI-main/lab1/compare_algorithms_6pts.py --games 100 --depth-low 2 --depth-high 4 --miss-chance 0.2 --seed 42


## Wyniki porownania

Parametry wspolne:
- liczba gier na konfiguracje: 100
- naprzemienny gracz rozpoczynajacy (AI1/AI2)
- AI1: glebokosc 2
- AI2: glebokosc 4

### Negamax z alpha-beta

1. Wariant deterministyczny
	- wygrane AI1: 0 (0.0%)
	- wygrane AI2: 50 (50.0%)
	- remisy: 50 (50.0%)
	- sredni czas ruchu AI1: 0.050 ms (400 ruchow)
	- sredni czas ruchu AI2: 0.346 ms (400 ruchow)

2. Wariant probabilistyczny (miss chance = 0.2)
	- wygrane AI1: 34 (34.0%)
	- wygrane AI2: 52 (52.0%)
	- remisy: 14 (14.0%)
	- sredni czas ruchu AI1: 0.056 ms (397 ruchow)
	- sredni czas ruchu AI2: 0.395 ms (401 ruchow)

### Negamax bez alpha-beta

1. Wariant deterministyczny
	- wygrane AI1: 0 (0.0%)
	- wygrane AI2: 50 (50.0%)
	- remisy: 50 (50.0%)
	- sredni czas ruchu AI1: 0.096 ms (400 ruchow)
	- sredni czas ruchu AI2: 2.558 ms (400 ruchow)

2. Wariant probabilistyczny (miss chance = 0.2)
	- wygrane AI1: 34 (34.0%)
	- wygrane AI2: 52 (52.0%)
	- remisy: 14 (14.0%)
	- sredni czas ruchu AI1: 0.111 ms (397 ruchow)
	- sredni czas ruchu AI2: 2.805 ms (401 ruchow)


## Wnioski (obowiazkowe minimum)

1. Wieksza glebokosc (4) daje przewage nad mniejsza (2) w obu wariantach gry.
2. Odciecie alpha-beta wyraznie przyspiesza decyzje AI (szczegolnie dla glebokosci 4).
3. Wariant probabilistyczny zmniejsza liczbe remisow i zwieksza losowosc wyniku,
	ale przewaga silniejszego AI pozostaje widoczna.
