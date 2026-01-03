# Analiza si Clasificarea Productiei de Energie

Acest proiect are ca scop clasificarea nivelului de productie a energiei (Low, Medium, High) pe baza datelor meteorologice si temporale, utilizand algoritmi de Invastare Automata (Machine Learning) in Python.

## Descriere Proiect

Setul de date utilizat (Energy Production Dataset.csv) contine informatii despre sursele de energie (Solar/Eolian), sezonalitate, ora zilei si cantitatea de energie produsa.

Obiectivul principal a fost transformarea problemei intr-una de clasificare supervizata, impartind productia in trei categorii (Tertile):
1. Low (Productie scazuta)
2. Medium (Productie medie)
3. High (Productie ridicata)

Etapele parcurse:
1. Analiza Exploratorie (EDA): Identificarea distributiilor si curatarea datelor.
2. Preprocesare:
   - Discretizarea variabilei tinta (Production).
   - Codificarea variabilelor categorice (One-Hot Encoding).
   - Scalarea standard a datelor (pentru algoritmi sensibili la distante).
3. Modelare: Antrenarea si compararea a 6 algoritmi diferiti conform metodologiei de laborator.

## Tehnologii Utilizate

- Limbaj: Python 3.x
- Manipulare Date: Pandas, NumPy
- Machine Learning: Scikit-Learn (Decision Trees, k-NN, Logistic Regression, LDA, QDA, Naive Bayes)
- Vizualizare: Matplotlib, Seaborn

## Rezultate si Performanta

In urma testelor, s-a observat ca setul de date prezinta relatii neliniare puternice, motiv pentru care modelele liniare au avut performante modeste.

Clasamentul Modelelor (dupa Acuratete):

1. Decision Tree (Fara limita) - 68.27% (Cel mai bun model)
2. Decision Tree (Depth=4) - 51.18%
3. Decision Tree (Depth=3) - 50.85%
4. k-NN (k=7) - 49.66%
5. Linear Discriminant Analysis (LDA) - 47.92%
6. Logistic Regression - 48.00%

Concluzii Tehnice:
Arborele de Decizie a depasit celelalte modele, demonstrand ca productia de energie depinde de combinatii specifice de factori, reguli pe care modelele liniare nu le pot invata eficient.

## Instructiuni de Rulare

1. Se cloneaza repository-ul:
   git clone https://github.com/Petrisor29/Energy-Production-Classification.git

2. Se instaleaza bibliotecile necesare:
   pip install -r requirements.txt

3. Se ruleaza scriptul de clasificare:
   python clasificare.py

