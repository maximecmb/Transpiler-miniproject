# Documentation du Transpileur de Modèles ML vers C

## Utilisation

### Options disponibles
- `--model, -m`: Chemin vers le fichier .joblib (par défaut: regression.joblib)
- `--out-c, -o`: Chemin du fichier C généré (par défaut: generated_linear_model.c)
- `--compile`: Compile le fichier C généré avec gcc
- `--run`: Exécute le binaire compilé (implique --compile)
- `--compare`: Compare la prédiction du binaire à model.predict()
- `--features`: Liste de features pour le test (ex: --features 1.0 2.0 3.0)

### Exemples

#### Régression linéaire
```bash
python transpile_simple_model.py \
  --model regression.joblib \
  --out-c linear_model.c \
  --compile --run --compare
```

#### Régression logistique
```bash
python transpile_simple_model.py \
  --model logistic_regression.joblib \
  --out-c logistic_model.c \
  --compile --compare \
  --features 1.0 1.0 1.0
```

#### Arbre de décision
```bash
python transpile_simple_model.py \
  --model decision_tree.joblib \
  --out-c tree_model.c \
  --compile --run --compare
```
