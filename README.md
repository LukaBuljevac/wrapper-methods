# Wrapper feature selection projekt - binarna klasifikacija

Ovo je pojednostavljena verzija projekta prema konzultacijama:

- datasetovi: za sada `parkinsons` i `hill_valley`
- klasifikatori: `knn5` i `svm_rbf`
- metode: baseline, SFS, SBS, random wrapper
- selekcijski kriteriji: accuracy i macro-F1
- protokol: 30 stratificiranih train/validation/test podjela
- omjer podjele: 50% train, 25% validation, 25% test
- izlaz: CSV rezultati, deskriptivna statistika i grafovi

## 1. Instalacija

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instalacija paketa:

```bash
pip install -r requirements.txt
```

## 2. Preuzimanje datasetova

```bash
python prepare_datasets.py
```

Skripta koristi UCI repozitorij i sprema:

```text
datasets/parkinsons.csv
datasets/hill_valley.csv
```

Ako preuzimanje ne radi, CSV datoteke možeš ručno staviti u mapu `datasets/`. Target stupac mora biti zadnji ili se mora zadati preko `--target-column`.

## 3. Brzi test

Ovo je za provjeru da sve radi:

```bash
python run_experiments.py --splits 2 --random-trials 20 --classifiers knn5
python plot_results.py
```

## 4. Eksperiment prema konzultacijama

```bash
python run_experiments.py --datasets parkinsons,hill_valley --classifiers knn5,svm_rbf --splits 30 --random-trials n2
python plot_results.py
```

`--random-trials n2` znači da se za dataset s `n` featurea generira `n*n` random podskupova.

Primjer:

- 22 featurea -> 484 random podskupa
- 100 featurea -> 10000 random podskupova

To može trajati dugo, osobito za SVM i Hill-Valley.

## 5. Što se sprema

U mapi `results/` dobiješ:

- `all_results.csv` - svi rezultati za svaki split
- `summary_results.csv` - prosjek, standardna devijacija, minimum i maksimum
- PNG grafovi

## 6. Kako protokol radi

Za svaki dataset, klasifikator i split:

1. Podaci se dijele na train/validation/test = 50/25/25.
2. Koristi se `stratify=y`, tako da omjer klasa ostane očuvan.
3. Baseline koristi sve featuree i trenira se bez selekcije.
4. SFS, SBS i random wrapper biraju featuree na train/validation dijelu.
5. Nakon odabira featurea, finalni model se trenira na train+validation i testira na test skupu.
6. Spremaju se accuracy, macro-F1, broj odabranih featurea i vrijeme.

## 7. Metode

### Baseline

Nema odabira značajki. Koriste se sve značajke.

### SFS

Kreće od praznog skupa i dodaje značajke dok god se validation score poboljšava.

### SBS

Kreće od svih značajki i uklanja značajke dok god se validation score poboljšava.

### Random wrapper

Generira nasumične podskupove značajki i bira onaj koji ima najbolji validation score.

## 8. Evaluacijske metrike

Na test skupu se uvijek računaju:

- accuracy
- macro-F1

Kod selekcije se zasebno rade dvije verzije:

- odabir prema accuracy
- odabir prema macro-F1

Zato postoje metode poput:

- `sfs + accuracy`
- `sfs + macro_f1`
- `sbs + accuracy`
- `sbs + macro_f1`
- `random + accuracy`
- `random + macro_f1`
