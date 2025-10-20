"""Script de test pour créer un modèle d'arbre de décision"""
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
import numpy as np

# Créer un jeu de données simple
X, y = make_regression(n_samples=100, n_features=4, n_informative=3,
                       noise=10, random_state=42)

# Entraîner un arbre de décision (limité en profondeur pour le test)
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X, y)

# Sauvegarder le modèle
joblib.dump(model, 'decision_tree.joblib')

print("Modèle d'arbre de décision créé et sauvegardé.")
print(f"Nombre de features: {model.n_features_in_}")
print(f"Nombre de noeuds: {model.tree_.node_count}")
print(f"Profondeur: {model.get_depth()}")

# Test avec un exemple
x_test = np.array([1.0, 1.0, 1.0, 1.0])
prediction = model.predict([x_test])[0]
print(f"\nTest avec x=[1, 1, 1, 1]:")
print(f"Prédiction: {prediction:.6f}")

