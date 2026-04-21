# Heart Disease Classification

Projet de classification binaire du dataset UCI Heart Disease avec plusieurs modeles de machine learning.

## Fichiers principaux

- `heart_disease_classification.ipynb`: notebook complet (EDA, preprocessing, modeles, comparaison finale)
- `heart_disease_classification.py`: version script
- `heart_disease_portfolio.ipynb`: version orientee presentation
- `Heart Disease (4).csv`: dataset source

## Algorithmes utilises

- Naive Bayes (Gaussian, Categorical, Mixed)
- Regression Logistique
- k-NN
- SVM (lineaire et noyaux)
- LDA
- Arbre de Decision

## Installation locale

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer le notebook

```bash
jupyter notebook
```

## Lancer le script

```bash
python heart_disease_classification.py
```

## Publication GitHub

1. Creer un nouveau repository vide sur GitHub (par exemple `heart-disease-classification`).
2. Initialiser git en local et pousser:

```bash
git init
git add .
git commit -m "Initial commit: heart disease classification project"
git branch -M main
git remote add origin https://github.com/<username>/heart-disease-classification.git
git push -u origin main
```

3. Ajouter ensuite les mises a jour:

```bash
git add .
git commit -m "Update notebook and results"
git push
```

## Notes

- Le rendu Graphviz PNG de l'arbre demande l'installation systeme de Graphviz (`dot` dans le PATH).
- Les sorties generees sont placees dans `outputs/`.
