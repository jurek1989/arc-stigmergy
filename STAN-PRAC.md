# Stan prac — punkt wznowienia

*Plik roboczy. `ARC-KONTEKST.md` = dlaczego robimy ten projekt, `CLAUDE.md` = jak jest
zorganizowany kod, ten plik = gdzie dokładnie stanęliśmy i co dalej.*

**Ostatnia aktualizacja: 31 lipca 2026.** Zapisane przed restartem maszyny.

---

## 1. Gdzie jesteśmy

| Etap | Stan |
| --- | --- |
| **Etap 0** — rusztowanie pomiarowe | **Ukończony**, zacommitowany, wypchnięty |
| **Etap 0.5** — pomiar dynamiki biblioteki | **W trakcie**, punkty 1–2 z 7 gotowe |
| **Etap 1** — mechanizm | **Nie zaczęty. Granica twarda.** Żadnego DSL-a, prymitywów, gramatyki, przeszukiwania, kompresji |

Repozytorium: **https://github.com/jurek1989/arc-stigmergy** — publiczne, konto `jurek1989`,
branch `main`. Publiczne dlatego, że Jurek udostępnia link do burzy mózgów z Fablem 5 na
czacie (czat lepszy do burzy mózgów niż Claude Code).

Katalog roboczy: `/mnt/c/Users/jerzy/Projects/ARC-AGI` (WSL/Ubuntu).

---

## 2. Co jest zrobione — Etap 0 (zacommitowane)

Trzy commity: `02848c1`, `21c8cc6`, `bb8c5dc`.

```
arc/task.py       Task, Pair, Grid, load_split, find_task, grid_key, grids_equal
arc/metrics.py    compare_grids -> GridComparison, diff_mask, comparison_rank
arc/viz.py        plot_task, plot_tasks, plot_comparison, save, CLI
arc/harness.py    Solver protocol, evaluate -> RunResult, TaskResult, TestOutcome
arc/baselines.py  IdentitySolver, MostCommonTrainOutputSolver, RandomSymmetrySolver, D4_TRANSFORMS
scripts/run_baselines.py
tests/            101 testów, ~11 s z danymi, 86 przechodzi + 15 pomija bez danych
```

### Liczby, które trzeba pamiętać

Podłogi baseline'ów na `arc-agi-2/evaluation` (120 zadań) — **są przygwożdżone testem** w
`tests/test_baselines.py`, więc jeśli pękną, aktualizujemy `CLAUDE.md` w tym samym commicie:

| solver | official | shape acc | cell acc | hist dist |
| --- | --- | --- | --- | --- |
| identity | 0,0000 | **0,7083** | **0,8097** (86 zadań) | 0,2184 |
| most-common-train-output | 0,0000 | 0,275 | 0,431 | 0,4153 |
| random-symmetry | 0,0000 | 0,679 | 0,673 | 0,2184 |

Dwie rzeczy z tego wynikające:
1. **Solver zwracający wejście bez zmian ma `cell_accuracy` 0,81.** Każda miara częściowa
   ma sens tylko względem tej podłogi.
2. `identity` i `random-symmetry` mają **identyczną** odległość histogramów — D4 permutuje
   komórki, więc zachowuje histogram dokładnie. Darmowy test spójności metryki.

Na `arc-agi-2/training` (1000 zadań) podłoga **nie jest zerowa**: `most-common-train-output`
robi 8/1000 (0,8%), `random-symmetry` 1/1000. Nie uznawać pierwszych 1–2% za sygnał.

### Dane — pułapka, o której trzeba pamiętać

`data/` jest gitignorowane. Prowenancja w `data/README.md`.

**ARC-AGI-1 NIE jest niezależnym zbiorem kontrolnym.** 767 z 1000 zadań
`arc-agi-2/training` to zadania z ARC-AGI-1 z przetasowanymi parami. Sześć zadań siedzi
w **obu** zbiorach ewaluacyjnych: `0934a4d8`, `136b0064`, `16b78196`, `981571dc`,
`aa4ec2a5`, `da515329`. To 5 punktów procentowych na 120-zadaniowym evalu. Stąd
`drop_leaked=True` w `load_split` / `evaluate`.

---

## 3. Etap 0.5 — zadanie w toku

### Motywacja (ustalona, do zapisania w CLAUDE.md przy domykaniu etapu)

Rozstrzygający eksperyment tego projektu to **zestaw krzywych z całego przebiegu**
mechanizmu, a nie pojedynczy przebieg ewaluacyjny.

`ARC-KONTEKST.md` §6 mówi „**jeden wykres**: rozmiar biblioteki w funkcji liczby
rozwiązanych zadań". **To zdanie jest nieaktualne i trzeba je poprawić.** Powód: płaska
biblioteka przy płaskiej skuteczności to **kolaps**, a nie zbieżność — feromon wyparował,
wszyscy agenci chodzą tym samym martwym śladem, system po cichu przestał eksplorować.
Zbieżność i kolaps dają *identyczną* płaską krzywą. Jedna krzywa ich nie odróżni.

Dlatego przyrząd śledzi **cztery sygnały na jednej wspólnej osi compute**:

1. skuteczność vs compute,
2. rozmiar biblioteki,
3. entropia użycia prymitywów,
4. długość opisu nowo przyjętych rozwiązań (musi spadać, jeśli kompresja działa).

### Punkt 1 — `arc/dynamics.py` — **GOTOWE** (niezacommitowane w chwili pisania)

`EpochSnapshot` (frozen dataclass) + `DynamicsLog` + trzy statystyki reżimowe.

Pola snapshotu: `epoch`, `compute` (kumulatywne), `tasks_solved` (kumulatywne, distinct),
`library_size`, `usage_counts` (mapping id → count), `new_description_lengths` (tuple,
może być pusta).

Pochodne: `usage_entropy_bits`, `usage_entropy_bits_miller_madow`,
`normalized_usage_entropy`, `mean_new_description_length`, `total_usage`,
`n_distinct_used`, `n_new_solutions`.

Statystyki reżimowe (w tym samym module, bo to część przyrządu, nie testów):
`library_growth_ratio(log, fraction=0.25)`, `description_length_trend(log)`,
`final_normalized_entropy(log)`.

### Punkt 2 — semantyka compute — **GOTOWE**

Sekcja „Compute" w `CLAUDE.md`, jedna definicja dla całego projektu:
**compute = liczba kandydujących programów ocenionych względem par treningowych zadania.**
Ta sama wielkość, którą harness zapisuje jako `TaskResult.steps` i którą
`EpochSnapshot.compute` kumuluje. Docstringi odsyłają tam zamiast powtarzać.

### Punkty 3–7 — **DO ZROBIENIA**

**3. Wykresy.** `plot_dynamics(log) -> Figure`, cztery panele na wspólnej osi compute.
Konwencje jak w `viz.py`: funkcja zwraca `Figure`, nic nie jest pokazywane ani zapisywane
bez proszenia, plus małe CLI `python -m ... --log results/<file>.json --out out.png`.

*Decyzja podjęta, do wykonania:* mieszka w **nowym module `arc/dynamics_viz.py`**, nie
w `viz.py`. Powód: dziedziną `viz.py` są **siatki** (paleta ARC, rysowanie komórek,
nakładki diff); wykresy dynamiki to szeregi czasowe, które nie dzielą z tym niczego poza
matplotlibem, a wrzucenie ich do `viz.py` zmusiłoby renderer siatek do zależności od
`dynamics.py`. Rozdzielone: `arc.viz` = „oglądaj zadania", `arc.dynamics_viz` =
„oglądaj przebieg".

*Ustalone z Jurkiem (odpowiedź na pytanie):* oprócz `plot_dynamics(log)` ma powstać
**`plot_dynamics_comparison(logs: Mapping[str, DynamicsLog])`** — cztery panele, w każdym
trzy kolorowe linie, **wspólne osie**. To jest forma, w której renderujemy trzy reżimy
(punkt 5), bo o odróżnialność reżimów w tym obrazku chodzi, a przy identycznych skalach
widać ją od razu.

Szczegóły paneli:
- panel 1: `tasks_solved` vs `compute`
- panel 2: `library_size` vs `compute`
- panel 3: `normalized_usage_entropy` linią ciągłą. **W widoku pojedynczego loga** dorysować
  drugą, bladą linię z entropią Millera–Madowa, żeby rozjazd był widoczny. W widoku
  porównawczym tylko plug-in (3 logi × 2 linie = kasza).
- panel 4: `mean_new_description_length` per epoka; epoki bez przyjęć to **przerwy**, nie
  zera (stąd `DynamicsLog.defined_series`). W widoku pojedynczego loga dorzucić blady
  scatter pojedynczych DL-i.
- oś x: `compute` kumulatywne, skala liniowa.

**4. Kalibracja przez syntetyczne reżimy** — analogon tego, czym `baselines.py` jest dla
harnessa. Trzy deterministyczne, zaseedowane generatory zwracające kompletne `DynamicsLog`.
*Decyzja:* mieszkają w **`arc/regimes.py`** (bo `baselines.py` też jest modułem w `arc/`).

Zaprojektowane sygnatury (do zaimplementowania):

| | `healthy` | `noise` | `collapse` |
| --- | --- | --- | --- |
| `tasks_solved` | rośnie, nasyca się wysoko | pełza | plateau nisko |
| `library_size` | wyraźnie **podliniowo** | **liniowo** | plateau wcześnie |
| entropia znormalizowana | umiarkowana, stabilna (0,5–0,85) | **wysoka** (>0,85), użycie rozmyte po rozdętej bibliotece | **spada do ~0** (<0,2), użycie zbiega na garstkę |
| DL nowych rozwiązań | **trend w dół** | mniej więcej płaski | spada trochę, potem płasko; w późnych epokach **puste listy** (nic nie jest przyjmowane — celowo ćwiczy ścieżkę pustej listy) |

Testy mają przygwoździć liczbowe sygnatury odróżniające trzy reżimy, np.:
- porządek końcowych entropii znormalizowanych: `noise > healthy > collapse`
- `library_growth_ratio`: healthy < 0,5; noise ≈ 1 (w [0,8; 1,25]); collapse < 0,1
- znak `description_length_trend`: healthy wyraźnie ujemny; noise ≈ 0; healthy najbardziej
  ujemny z trójki
- `final tasks_solved`: healthy > noise, healthy > collapse

Sens: **każda przyszła zmiana, która rozmyje zdolność przyrządu do odróżnienia reżimów,
ma wywalić test.**

**5. `scripts/plot_dynamics_demo.py`** — renderuje trzy reżimy do jednego PNG (nałożone na
wspólnych osiach, patrz punkt 3), żeby człowiek zobaczył gołym okiem, że są odróżnialne.

**6. Testy** — szybkie, bez zależności od danych zadaniowych. Pokryć: przypadki brzegowe
entropii (jeden prymityw, użycie równomierne), round-trip serializacji, walidację
monotoniczności, sygnatury reżimów. Całość suite ma zostać wygodnie interaktywna.

**7. Aktualizacja `CLAUDE.md`** — sekcja o Etapie 0.5, tabela layoutu, komendy uruchamiania,
semantyka compute/steps (już jest), notka o kalibracji trójreżimowej. W istniejącym tonie,
bez marketingu.

**Plus, ustalone osobno:** poprawić `ARC-KONTEKST.md` §6, żeby dwa dokumenty nie mówiły
sprzecznych rzeczy o „jednym wykresie". Jurek zgodził się na aktualizację (nie zostawiamy
tego jako zapisu stanu z lipca).

---

## 4. Decyzje projektowe podjęte w Etapie 0.5 (i dlaczego)

**Entropia — liczymy OBA estymatory.** Ustalone z Jurkiem.
- plug-in: `H = -Σ pᵢ log₂ pᵢ`, `pᵢ = cᵢ / N`
- Miller–Madow: `H_MM = H + (K-1)/(2N ln2)`, K = liczba prymitywów z `cᵢ > 0`, N = `Σ cᵢ`

Powód: plug-in jest **obciążony w dół** o ok. `(K-1)/(2N ln2)` bitów. We wczesnych epokach
N jest małe, a K rośnie, więc obciążenie jest największe dokładnie tam, gdzie odróżniamy
kolaps — zdrowa, szeroko używana młoda biblioteka czytałaby się jak kolaps.

**`normalized_usage_entropy` normalizuje plug-in, nie MM.** Konkretny dowód, że to
konieczne: dla równomiernego użycia 4 prymitywów przy N=4 plug-in daje `H = 2,000` bitu,
a MM `2,541` — **powyżej `log2(4) = 2`**, czyli powyżej maksimum teoretycznego. Plug-in
jest ograniczony przez `log2(library_size)`, więc iloraz zostaje w [0,1]; MM nie ma takiego
ograniczenia i dawałby ilorazy > 1. MM leży obok jako surowa liczba w bitach, a **rozjazd
między nimi jest sam w sobie diagnostyką**: dopóki jest szeroki, korpus jest za mały, żeby
w ogóle czytać panel entropii.

**`library_size` NIE jest walidowane jako monotoniczne.** Kryterium kompresji, które
również *usuwa* prymitywy, jest całym sednem — log zabraniający kurczenia się nie mógłby
zapisać parowania feromonu. Za to `epoch` i `compute` muszą rosnąć **ściśle** (epoka, która
nie spaliła compute, to nie epoka; powtórzony indeks znaczy, że dwóch pisarzy dopisuje do
tego samego loga), a `tasks_solved` nie może maleć.

**`usage_counts` może pokrywać mniej prymitywów niż `library_size`** (prymityw może leżeć
w bibliotece nieużywany), ale nigdy więcej — sprawdzane, rzuca wyjątek.

**`mean_new_description_length` to `None`, nie 0.0**, gdy epoka niczego nie przyjęła. Ta
sama zasada co przy `cell_accuracy` w `metrics.py`: epoka, która nic nie przyjęła, to nie
epoka, która przyjęła coś za darmo.

**`usage_counts` zamrożone w `MappingProxyType`** — spójnie z read-only siatkami.

**CSV = tylko skalary pochodne, JSON = pełne dane.** Mapa użyć nie mieści się w komórce
CSV. Ten sam podział ról co w `RunResult.save`: JSON to archiwum i jedyne, co czyta
`load()`, CSV to płaska tabela do Polarsa.

**Nazwa pliku: `<stem>.dynamics.json` / `.dynamics.csv`**, gdzie stem jest **identyczny**
jak w `RunResult.save`: `<timestamp>__<slug(name)>__<slug(dataset)>_<slug(split)>`. Dzięki
temu plik z wynikiem i plik z dynamiką tego samego przebiegu leżą obok siebie i się nie
zderzają. `arc/dynamics.py` ma własne `_slug` (nie importuje prywatnego z `harness.py`);
**do napisania test przygważdżający, że oba dają identyczny stem.**

**`dynamics.py` nie zawiera żadnego mechanizmu.** Przyjmuje gołe liczby. Przyszłe
przeszukiwanie samo zbuduje `EpochSnapshot`. Celowo nie ma tam abstrakcyjnego interfejsu
pod mechanizm, który nie istnieje.

---

## 5. Konwencje projektu, o których nie wolno zapomnieć

- **Siatki są read-only `numpy.uint8`.** Zadania są cache'owane per proces, więc mutacja
  zatrułaby późniejsze przebiegi. Read-only zamienia to w wyjątek w miejscu błędu.
- **`ndarray` nie jest hashowalny, a `==` jest element-wise.** `grid_key(grid)` do kluczy
  słownika, `grids_equal(a, b)` do równości.
- `Task` i `Pair` to frozen dataclass z `eq=False` (wygenerowane `__eq__` po polach ndarray
  rzuca).
- Kolor 0 = tło. Konwencja ARC, nie gwarancja; `foreground_iou` na tym stoi.
- Kontrakt solvera: `solve(task) -> list[list[Grid]]` — jedna pozycja na wejście testowe,
  każda 1–2 kandydujące siatki. `steps` opcjonalne, czytane przez `getattr` po `solve`.
- **Kod, komentarze, docstringi i commit messages po angielsku.** Rozmowa z Jurkiem po
  polsku.
- Numpy, matplotlib, biblioteka standardowa. Żadnych frameworków ML, żadnych wywołań
  sieciowych z kodu projektu, żadnej rejestracji nigdzie.
- **Nie budujemy na zapas.** Jeśli decyzja projektowa jest podjęta pod przyszły etap, ma to
  być powiedziane wprost i uzasadnione.

### Styl pracy, o który Jurek prosił

- Pytać, zanim zaimplementuje się coś, co ma więcej niż jedno sensowne odczytanie.
- Mówić wprost, jeśli jego decyzja wygląda na błędną albo jeśli coś jest projektowane pod
  problem, którego nie będzie.
- **Podsumowanie po każdym ukończonym punkcie, nie po całym etapie.**
- Nie dodawać funkcjonalności, o które nie prosił.
- Sparring partner, nie klakier. Nie chwalić za oczywistości.

---

## 6. Środowisko

- WSL/Ubuntu na Windowsie, Python **3.14.4**, numpy **2.4.6**, matplotlib **3.11.0**,
  pytest **9.1.1**. Wszystko systemowe, bez venv.
- Repo na `/mnt/c` — filesystem WSL-a jest wolny; wczytanie 1920 plików JSON zajmuje ~8,5 s
  i **nie ma tam czego optymalizować**. Pełny suite ~11 s; przy pracy nad mechanizmem
  odpalać `python -m pytest tests/test_harness.py` (natychmiastowe, zero danych).
- `gh` CLI: **aktywne konto przełączone na `jurek1989`** (było `jszocik`). Zostaje tak.
- `user.name` / `user.email` ustawione **lokalnie w tym repo** (globalnej konfiguracji gita
  nie ma i nie ruszaliśmy jej).
- Autoryzacja Claude Code: `claude.ai`, subskrypcja — **nie** klucz API. Sprawdzać na
  starcie każdej sesji (`claude auth status`).

### Komendy

```bash
python scripts/run_baselines.py                       # baseline'y na arc-agi-2/evaluation
python scripts/run_baselines.py --limit 20 --no-save  # smoke test
python -m arc.viz --task 0934a4d8 --out task.png
python -m arc.viz --split evaluation --limit 24 --out sheet.png
python -m pytest                                      # 101 testów, ~11 s
python -m pytest tests/test_harness.py                # bez danych, natychmiast
```

---

## 7. Jak wznowić

1. `claude auth status` — potwierdzić `claude.ai`, nie `api_key`.
2. Przeczytać `ARC-KONTEKST.md`, `CLAUDE.md` i ten plik.
3. `python -m pytest` — powinno być **101 passed**. Jeśli `data/` zniknęło, będzie
   *86 passed, 15 skipped* i trzeba pobrać dane wg `data/README.md`.
4. Ruszyć od **punktu 3** sekcji „Etap 0.5": `arc/dynamics_viz.py`, potem `arc/regimes.py`
   (punkt 4), `scripts/plot_dynamics_demo.py` (punkt 5), testy (punkt 6), `CLAUDE.md` +
   `ARC-KONTEKST.md` §6 (punkt 7).
5. **Nie zaczynać Etapu 1.** Granica jest twarda: żadnego DSL-a, prymitywów, gramatyki,
   przeszukiwania ani kompresji.

---

## 8. Wątki otwarte (z ARC-KONTEKST.md §7, wciąż aktualne)

- Mechanizm stygmergii **formalnie** — definicja medium i reguły parowania, żeby to nie
  było tylko metaforą.
- **Kryterium kompresji z DreamCodera** — matematyka wyboru abstrakcji.
- **Sztuczka MDL z CompressARC** — VAE loss jako zamiennik przeszukiwania kombinatorycznego.
- Mechanizm **StochasticGoose** — eksploracja na sygnale „coś się zmieniło".
- Związek z równoległym projektem o automatach komórkowych (Life → Lenia → Neural CA).

Jurek prowadzi równolegle burzę mózgów z Fablem 5 na czacie i wróci z pomysłami. Repo jest
publiczne właśnie po to, żeby Fable miał dostęp do kodu.
