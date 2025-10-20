"""Script de test pour créer un modèle de régression logistique"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np

# Créer un jeu de données simple
X, y = make_classification(n_samples=100, n_features=3, n_informative=2,
                          n_redundant=1, random_state=42)

# Entraîner un modèle de régression logistique
model = LogisticRegression(random_state=42)
model.fit(X, y)

# Sauvegarder le modèle
joblib.dump(model, 'logistic_regression.joblib')

print("Modèle de régression logistique créé et sauvegardé.")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Test avec un exemple
x_test = np.array([1.0, 1.0, 1.0])
proba = model.predict_proba([x_test])[0, 1]
print(f"\nTest avec x=[1, 1, 1]:")
print(f"Probabilité (classe 1): {proba:.6f}")

