# ARC Prize — plik kontekstowy projektu

*Podsumowanie rozmowy założycielskiej. Czytaj to na wejściu do każdej nowej dyskusji w tym projekcie.*

---

## 1. Kim jestem i po co to robię

Jurek — matematyk z wykształcenia, data scientist przechodzący w AI engineering. Pracuję sam, w wolnych chwilach, na ultrabooku (WSL/Ubuntu), z Claude Code jako narzędziem implementacyjnym.

**To nie jest projekt zarobkowy.** Nie startuję po nagrody, nie ścigam się z Tufa Labs ani z NVIDIĄ. Cel to: (a) ciekawa łamigłówka na wolny czas, (b) samorozwój, w szczególności po stronie systemów agentowych, (c) sprawdzenie na żywo dwóch własnych pomysłów badawczych.

**Czego świadomie NIE robię:** nie idę drogą „wygeneruj masę syntetycznych danych w stylu ARC i dotrenuj model". To działa i daje wyniki, ale zaprzecza pierwotnej idei benchmarku i mnie po prostu nie interesuje. Nie lubię też szlifowania inżynierskich detali — interesuje mnie wymyślanie i testowanie pomysłów.

Preferowany tryb rozmowy: sparring partner. Dopytuj, precyzuj, podważaj. Nie chwal za oczywistości.

---

## 2. Krajobraz konkursu (stan: lipiec 2026)

**ARC Prize 2026**, ~$2M w trzech torach. Start 25 marca 2026, submissions do 2 listopada, papery do 8 listopada, wyniki 4 grudnia. Warunek konieczny każdej nagrody: **open source na licencji public-domain (CC0/MIT-0), otwarte *przed* poznaniem oficjalnych wyników**. Ewaluacja offline na Kaggle — **żadnych zewnętrznych API**.

### Tor ARC-AGI-2 (statyczne siatki) — $700k
- Dwie próby na zadanie, dopasowanie siatki **co do piksela**, zero-jedynkowo.
- $275k progress prizes (top 8), $275k Grand Prize za najlepszy *writeup* (oceniany rubryką, subiektywnie), $150k bonus za ≥85% (praktycznie na pewno przechodzi na 2027).
- Wyniki 2025 pod ograniczeniami Kaggle: 1. NVARC 24,03%, 2. ARChitects 16,53%, 3. MindsAI 12,64%. W 2026 publiczne notebooki chodzą już w okolicach 30–34%.
- Uwaga na dwa różne leaderboardy: publiczny (frontier API, GPT-5.5 ~85%) vs Kaggle (offline, ~$50 compute na submission). To zupełnie inne gry.

### Tor ARC-AGI-3 (interaktywny) — $850k
- Setki ręcznie zaprojektowanych gier turowych. **Brak instrukcji, reguł i podanego celu.** Agent musi sam odkryć mechanikę i to, co znaczy „wygrać".
- 8–10 poziomów na grę, **każdy kolejny poziom wprowadza nową mechanikę**.
- Zbiory: 25 publicznych demo, 55 semi-prywatnych (leaderboard), 55 w pełni prywatnych (konkurs).
- **Metryka to efektywność akcji względem człowieka:** per poziom `(człowiek/agent)²` z sufitem 1,15, poziomy ważone indeksem, wynik per gra capowany na 100. Baseline z kontrolowanego badania >200 osób, liczony z *pierwszego* przebiegu gracza.
- Wyniki startowe: ludzie 100%, frontier AI 0,51% (GPT-5.4, Claude Opus 4.6, Grok 4.2 w przedziale 0–0,37%). Najlepszy dedykowany agent z preview: 12,58%.
- Milestone #2 zamyka się 30 września.

### Paper Track
Submission z kodem **nie musi osiągnąć wysokiego wyniku**, żeby paper był kwalifikowany. Nagradzają ideę, nie ranking. Precedensy: CompressARC (76k parametrów, bez pretreningu, MDL per zadanie, RTX 4070, 3. miejsce), ARC-NCA (Neural Cellular Automata, runner-up).

---

## 3. Diagnoza: dlaczego ARC-AGI-2 stał się zawodami inżynierskimi

Kluczowa teza, do której doszliśmy — warto ją trzymać, bo tłumaczy wybór kierunku.

**Prywatny zbiór testowy chroni przed zapamiętaniem konkretnych zadań, ale nie przed zapamiętaniem *gatunku* zadań.** ARC jest publiczny od 2019: tysiąc zadań treningowych, dziesiątki repozytoriów, DSL-e, generatory, papery — wszystko wsiąkło w korpusy treningowe. Model nie widział Twojego zadania, ale zna format, paletę i repertuar transformacji. Rzeczy, które miały być testowane jako **wrodzone priory**, stały się **wyuczoną treścią**.

Dowód poszlakowy od organizatorów: Gemini 3 Deep Think używa poprawnych mapowań kolorów ARC w rozumowaniu, mimo że harness nie wspomina ani o ARC, ani o kolorach.

**Druga rzecz — pułapka IID.** Publiczny train i prywatny eval pochodzą z tego samego procesu generującego (ci sami projektanci, te same reguły, ta sama kalibracja trudności). Więc „nowe zadanie" znaczy w praktyce „nowa próbka ze znanego rozkładu", a uczenie się rozkładu z wielu próbek to standardowy ML. Benchmark miał mierzyć radzenie sobie ze spoza-doświadczenia, a mierzy interpolację w obrębie znanej rodziny.

**Konsekwencja:** przepis na wysoki wynik nie wymaga nowej idei — wymaga syntetycznych danych, test-time trainingu, ensemble'u augmentacji i dyscypliny pipeline'u. Diagnoza samych organizatorów: *luka w dokładności jest zablokowana głównie przez inżynierię, luka w efektywności wciąż wymaga nauki i nowych idei.*

**Kontrargument, który trzeba pamiętać:** człowiek też widział wcześniej mnóstwo łamigłówek i o nim nie mówimy „skontaminowany", tylko „poznał dziedzinę". Odpowiedź Cholleta to efektywność (człowiek nabył priory z żenująco małej liczby przykładów). Ale to przerzuca definicję na oś, której leaderboard nie mierzy. Głębsza wersja całości: **prawo Goodharta** — każda operacjonalizacja inteligencji jako stałego zbioru testowego zostaje z czasem wchłonięta przez rozkład treningowy.

**Uwaga do ARC-AGI-3:** teza „tych gier nie da się skontaminować" jest słabsza, ale nie zerowa. Kontaminacja przewędrowała z „formatu zadania" na **konwencje gier wideo** (że przycisk wygląda jak przycisk, że postać się porusza). Reki wprost zakodował te priory ręcznie; zamrożony LLM wnosi je z pretreningu.

---

## 4. Stan metagry w ARC-AGI-3 (Milestone #1, 6 lipca 2026)

1. **Tufa Labs „The Duck"** — mały open-source LLM (Qwen 3.6 27B FP8 lokalnie) grający przez pisanie i wykonywanie Pythona w żywym REPL-u. Stan gry jako zmienne Pythona. Percepcja przez obraz + surowy ASCII + segmentację, wybór reprezentacji ad hoc. „Nieskończona gra" przez wypychanie najstarszych wiadomości z kontekstu. **Ich własny wniosek: ręcznie robione narzędzia szkodziły modelowi, lepiej działała improwizacja.**
2. **Reki** — vision-LLM-jako-polityka (Gemma-4-31B lokalnie), jeden JSON z akcją na turę, pamięć refleksji co ~10 kroków, plus heurystyki klikania w numpy („klikaj w małe, rzadkokolorowe, przyciskopodobne kształty", „martwa sygnatura" wyłącza bezowocne typy obiektów).
3. **„forge"** — to samo co Reki w konfigurowalnym frameworku; **najlepszy przebieg używał profilu wyłączającego całą dodatkową maszynerię.**

Dwóch z trzech wystartowało z oficjalnego szablonu ARC Prize, podmieniając model. **Wniosek: obecna meta zsunęła się w harness engineering** — nie „jakie dane wygenerować", tylko „jak owinąć lokalny LLM promptem, pamięcią i pętlą".

**Szczelina, która zostaje otwarta.** W fazie preview, zanim pojawiły się szablony, wygrały rzeczy zupełnie inne i **żadna nie używała LLM-a**:
- *StochasticGoose* (12,58%) — CNN + proste RL przewidujące, **które akcje spowodują zmianę klatki**. Sygnał to nie nagroda, tylko „czy cokolwiek się wydarzyło". Gołe novelty-driven exploration. Koszt: 255 tys. akcji na 18 poziomów — czyli tryb brute-force, który nowa metryka ma dławić.
- *Blind Squirrel* (6,71%) — graf stanów z klatek, przycinanie akcji tworzących pętle, wsteczne etykietowanie odległościami i mały model wartości na ResNet18.

To są lekkie konstrukcje, rozwijalne na jednym GPU. **Tu leży realnie nieprzeorane pole: eksploracja napędzana motywacją wewnętrzną** (ciekawość, empowerment, novelty search, uczenie modelu świata bez nagrody).

---

## 5. Mój pomysł — stan obecny

### 5a. Co próbowałem przy ARC-AGI-2 (poprzednie podejście, nieukończone)

Budowa wstępnej **siatki pojęć** (obiekty + operacje + relacje), podanie jej lokalnemu LLM-owi (chyba Qwen), który miał oglądać zadanie i składać z tej siatki program. Przy porażce — rozszerzyć siatkę o jeden nowy element i spróbować ponownie.

**To trafia w żywy nurt badawczy.** W istocie jest to **DreamCoder** (Ellis i wsp.): indukcja programów w cyklu wake-sleep z rozbudową własnego DSL-a przez abstrahowanie powtarzających się podprogramów. Precedensy na ARC: SOAR (2. miejsce w paperach 2025, 52% na ARC-AGI-1 bez ręcznych DSL-i), podejście E. Panga (ewolucyjne przeszukiwanie w Pythonie z dynamicznie tworzoną biblioteką abstrakcji), ArcMemo.

### 5b. Zdiagnozowane wąskie gardło — i tu jest cała rzecz

Nie zatrzymała mnie umiejętność programowania. Zatrzymało mnie **kryterium wzrostu siatki**.

„Rozszerz o jeden nowy element" — o *który*? Przestrzeń możliwych prymitywów jest nieskończona i nieustrukturyzowana. Bez zasady wyboru dostajesz błądzenie losowe, a każda nieudana próba dokłada szum do biblioteki, przez co następne przeszukiwanie jest **trudniejsze**. Biblioteka rośnie, skuteczność spada. To klasyczna śmierć tego typu systemów.

**Odpowiedź DreamCodera: nie rozszerzaj przy porażce — rozszerzaj przy kompresji.** Rozwiąż paczkę zadań, przeszukaj znalezione programy za powtarzającymi się fragmentami i wyabstrahuj te, które **skracają łączny opis całego korpusu rozwiązań**. Nowy prymityw zasługuje na miejsce w bibliotece wtedy i tylko wtedy, gdy zmniejsza sumaryczną długość opisu (MDL). Kryterium obiektywne, mierzalne, bez zgadywania.

*Zastrzeżenie:* czysta synteza programów na ARC-AGI-2 uderzyła w ścianę — dwójka była projektowana tak, żeby zadania były kompozycyjne i opierały się stałemu DSL-owi. To nie jest droga do wysokiego wyniku. Jako obiekt badawczy — bardzo dobra.

### 5c. Drugi pomysł: inteligencja roju — i jego wada konstrukcyjna

Pierwotne sformułowanie: „dużo prostych agentów, jeden do wykrywania obiektów, drugi do wykonywania ruchu".

**To nie jest rój, to pipeline z podziałem funkcji.** Jeśli z góry decyduję, kto co robi, to **abstrakcję wykonuję ja, nie system** — struktura, która miała się wyłonić, została wpisana w warunki początkowe.

Co faktycznie robi robotę u mrówek:
1. **Stygmergia** — brak komunikacji między agentami; modyfikacja i odczyt **wspólnego środowiska**. Środowisko *jest* pamięcią i kanałem.
2. **Samowzmacniające się sprzężenie** — dobre rozwiązanie wzmacnia samo siebie fizycznie, nikt go nie ocenia.
3. **Jednorodność** — wszyscy mają tę samą regułę; podział pracy się *wyłania*, nie jest projektowany.

**Druga przeszkoda:** ARC-AGI-3 ma budżet akcji, a punktacja karze marnotrawstwo kwadratowo. Roje działają, gdy próby są tanie — tutaj akcje są dokładnie tym, czego się oszczędza. Rzucenie roju prosto w środowisko to strukturalna kolizja z regułami.

### 5d. Synteza — kierunek, który przyjmujemy

**Rój nie biega po środowisku. Rój biega po wyuczonym modelu świata.**

Agent buduje z klatek prosty model przejść (sygnał: co zmienia się po jakiej akcji — dokładnie to, na czym uczy się StochasticGoose). Wewnątrz tego modelu puszczamy tysiące tanich eksploratorów, którzy *halucynują*, a nie klikają. W realnym środowisku wykonujemy tylko akcje, co do których rój się skonsolidował — albo te, co do których się maksymalnie **nie** zgadza, bo tam model świata jest niepewny i jedna prawdziwa akcja da najwięcej informacji.

Wtedy równoległość roju kosztuje **compute, nie akcje**, a metryka efektywności staje się sprzymierzeńcem.

**I najważniejsze: to jest jeden pomysł, nie dwa.**

Siatka pojęć, do której wielu prostych poszukiwaczy dopisuje i z której odczytują, **jest śladem feromonowym**. Wspólne, zewnętrzne, trwałe medium, modyfikowane lokalnie przez agentów głupich z osobna i nieświadomych całości. To stygmergia w przestrzeni programów. **Kryterium kompresji z DreamCodera pełni rolę parowania feromonu:** ścieżki nieużywane zanikają, używane się wzmacniają.

---

## 6. Plan i zasady pracy

**Stół laboratoryjny: publiczny zbiór ewaluacyjny ARC-AGI-2, offline.** Nie zaczynamy od trójki. Powody prozaiczne: żadnych limitów API ani budżetu akcji, iteracja w sekundach, możliwość odpalenia setek przebiegów w nocy. Mechanizm, który tam zadziała, przeniesiemy do trójki; jak nie zadziała, dowiem się w tydzień, a nie w kwartał.

**Nie rejestrujemy się nigdzie na razie.** Dane bierzemy z publicznych repozytoriów. Submission to decyzja na wrzesień albo wcale.

**Minimalny eksperyment rozstrzygający:** populacja prostych agentów, wspólna biblioteka prymitywów, wzrost biblioteki **wyłącznie przez kompresję** korpusu rozwiązań, i **jeden wykres: rozmiar biblioteki w funkcji liczby rozwiązanych zadań**. Krzywa się wypłaszcza przy rosnącej skuteczności → mamy stygmergię. Biblioteka puchnie liniowo → mamy szum, trzeba zaostrzyć kryterium.

**Zidentyfikowane ryzyko numer jeden:** Claude Code usuwa wąskie gardło *implementacji*, nie *ewaluacji*. Ryzyko przesuwa się z „nie umiem tego zbudować" na „mam pięć wariantów w tydzień i nie wiem, który działa, bo nie zaprojektowałem pomiaru". **Pomiar budujemy przed mechanizmem.**

---

## 7. Otwarte wątki do rozebrania

- Mechanizm stygmergii **formalnie** — jak zdefiniować medium i regułę parowania, żeby to nie było tylko metaforą.
- **Kryterium kompresji z DreamCodera** — matematyka wyboru abstrakcji, domknięcie brakującego elementu z 5b.
- **Sztuczka MDL z CompressARC** — VAE loss jako zamiennik przeszukiwania kombinatorycznego; czysta teoria informacji.
- Mechanizm **StochasticGoose** — eksploracja na sygnale „coś się zmieniło" i dlaczego to w ogóle działa.
- Związek z moim równoległym projektem o automatach komórkowych (Life → Lenia → Neural CA): problem „jak system odkrywa strukturę środowiska bez zewnętrznej instrukcji" to w innym przebraniu problem emergencji.
