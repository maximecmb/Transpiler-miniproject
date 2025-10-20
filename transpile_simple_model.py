import argparse
import os
import subprocess
import sys
from typing import Any, Tuple

import joblib
import numpy as np


def find_estimator(obj: Any) -> Tuple[Any, str]:
    """
    Retourne (estimateur, type_modèle) où type_modèle est 'linear', 'logistic' ou 'tree'
    """
    # Pipeline-like
    if hasattr(obj, "steps"):
        for name, step in reversed(obj.steps):
            model_type = detect_model_type(step)
            if model_type:
                return step, model_type
        raise ValueError("Aucun estimateur supporté trouvé dans la pipeline.")
    # Direct estimator
    model_type = detect_model_type(obj)
    if model_type:
        return obj, model_type
    raise ValueError("Le modèle chargé n'est pas supporté.")


def detect_model_type(estimator: Any) -> str:
    """
    Détecte le type d'estimateur: régression linéaire, logistique ou arbre de décision
    """
    class_name = estimator.__class__.__name__

    # Arbres de décision
    if 'Tree' in class_name or 'DecisionTree' in class_name:
        if hasattr(estimator, 'tree_'):
            return 'tree'

    # Régressions avec coef_ et intercept_
    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
        if 'Logistic' in class_name:
            return 'logistic'
        elif 'Linear' in class_name or 'Ridge' in class_name or 'Lasso' in class_name:
            return 'linear'

    return None


def to_c_float_list(values, precision=8):
    """
    Formate une liste/ndarray de nombres Python en littéraux float C: 1.23456789f
    """
    return ", ".join(f"{float(v):.{precision}f}f" for v in values)


def generate_tree_c_source(tree, features_example: np.ndarray, prog_name: str, n_features: int) -> str:
    """
    Génère le code C pour un arbre de décision.
    Utilise l'attribut tree_ de scikit-learn.
    """
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    value = tree_.value
    children_left = tree_.children_left
    children_right = tree_.children_right

    feats_c = to_c_float_list(features_example)

    # En-tête
    header = f"""\
/*
 * Fichier généré automatiquement par {prog_name}.
 * Modèle: Arbre de décision
 * N_FEATURES = {n_features}
 * N_NODES = {tree_.node_count}
 */
#include <stdio.h>

#define N_FEATURES {n_features}

"""

    # Générer le code récursif pour parcourir l'arbre
    def generate_node_code(node_id, indent=1):
        indent_str = "    " * indent

        # Feuille: retourner la valeur
        if children_left[node_id] == children_right[node_id]:  # -1 == -1
            # Pour la régression, value est shape (1, 1, 1) généralement
            # Pour la classification, value est shape (1, n_samples, n_classes)
            # On prend la valeur majoritaire ou la moyenne
            if value[node_id].shape[-1] == 1:
                # Régression
                leaf_value = float(value[node_id].flatten()[0])
            else:
                # Classification: retourner la classe majoritaire (indice de la valeur max)
                class_counts = value[node_id].flatten()
                leaf_value = float(np.argmax(class_counts))
            return f"{indent_str}return {leaf_value:.8f}f;\n"

        # Noeud interne: condition sur une feature
        feat_idx = feature[node_id]
        thresh = threshold[node_id]

        code = f"{indent_str}if (features[{feat_idx}] <= {thresh:.8f}f) {{\n"
        code += generate_node_code(children_left[node_id], indent + 1)
        code += f"{indent_str}}} else {{\n"
        code += generate_node_code(children_right[node_id], indent + 1)
        code += f"{indent_str}}}\n"

        return code

    # Fonction de prédiction
    prediction_func = f"""float prediction(const float *features, int n_feature) {{
    if (n_feature != N_FEATURES) {{
        fprintf(stderr, "Erreur: n_feature=%d (attendu %d)\\n", n_feature, N_FEATURES);
        return 0.0f;
    }}
{generate_node_code(0).rstrip()}
}}
"""

    # Main de test
    main_func = f"""
int main(void) {{
    // Exemple de données statiques (modifiable)
    const float x[N_FEATURES] = {{ {feats_c} }};
    const float y = prediction(x, N_FEATURES);
    printf("y_hat=%.6f\\n", y);
    return 0;
}}
"""

    return header + prediction_func + main_func


def generate_c_source(intercept: float, coefs: np.ndarray, features_example: np.ndarray,
                      prog_name: str, model_type: str = 'linear') -> str:
    """
    Génère le code C complet : prédiction + main de test.
    Signature demandée : float prediction(float *features, int n_feature)
    model_type: 'linear' ou 'logistic'
    """
    n_feat = coefs.shape[0]
    thetas = np.concatenate(([intercept], coefs))
    thetas_c = to_c_float_list(thetas)
    feats_c = to_c_float_list(features_example)

    # En-tête commun
    model_desc = "régression logistique (y = sigmoid(theta0 + sum_i theta_i * x_i))" if model_type == 'logistic' \
                 else "régression linéaire (y = theta0 + sum_i theta_i * x_i)"

    header = f"""\
/*
 * Fichier généré automatiquement par {prog_name}.
 * Modèle: {model_desc}
 * N_FEATURES = {n_feat}
 */
#include <stdio.h>
#include <math.h>

#define N_FEATURES {n_feat}

static const float THETAS[1 + N_FEATURES] = {{ {thetas_c} }};
"""

    # Fonction sigmoid pour la régression logistique
    sigmoid_func = ""
    if model_type == 'logistic':
        sigmoid_func = """
static inline float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}
"""

    # Fonction de prédiction
    if model_type == 'logistic':
        prediction_func = """
float prediction(const float *features, int n_feature) {
    if (n_feature != N_FEATURES) {
        // Garde-fou simple : taille inattendue
        fprintf(stderr, "Erreur: n_feature=%d (attendu %d)\\n", n_feature, N_FEATURES);
        return 0.0f;
    }
    float z = THETAS[0]; // biais
    for (int i = 1; i <= N_FEATURES; ++i) {
        z += THETAS[i] * features[i - 1];
    }
    return sigmoid(z); // Probabilité entre 0 et 1
}
"""
    else:
        prediction_func = """
float prediction(const float *features, int n_feature) {
    if (n_feature != N_FEATURES) {
        // Garde-fou simple : taille inattendue
        fprintf(stderr, "Erreur: n_feature=%d (attendu %d)\\n", n_feature, N_FEATURES);
        return 0.0f;
    }
    float y = THETAS[0]; // biais
    for (int i = 1; i <= N_FEATURES; ++i) {
        y += THETAS[i] * features[i - 1];
    }
    return y;
}
"""

    # Main de test
    main_func = f"""
int main(void) {{
    // Exemple de données statiques (modifiable)
    const float x[N_FEATURES] = {{ {feats_c} }};
    const float y = prediction(x, N_FEATURES);
    printf("y_hat=%.6f\\n", y);
    return 0;
}}
"""

    return header + sigmoid_func + prediction_func + main_func


def main():
    parser = argparse.ArgumentParser(
        description="Transpile un modèle LinearRegression scikit-learn en code C d'inférence autonome."
    )
    parser.add_argument(
        "--model", "-m", default="regression.joblib",
        help="Chemin vers le fichier .joblib (par défaut: regression.joblib)"
    )
    parser.add_argument(
        "--out-c", "-o", default="generated_linear_model.c",
        help="Chemin du fichier C généré (par défaut: generated_linear_model.c)"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile le fichier C généré avec gcc (si disponible)."
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Exécute le binaire compilé (implique --compile)."
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare la prédiction du binaire à model.predict sur le même vecteur."
    )
    parser.add_argument(
        "--features", type=float, nargs="*",
        help="Liste de features pour le main C et la comparaison (ex: --features 1 1 1). "
             "Par défaut: un vecteur de 1.0 de la bonne taille."
    )
    args = parser.parse_args()

    # 1) Charger le modèle
    model_obj = joblib.load(args.model)
    est, model_type = find_estimator(model_obj)

    # 2) Déterminer le nombre de features et préparer les exemples
    if model_type == 'tree':
        n_features = est.n_features_in_
        if args.features is None:
            features_example = np.ones(n_features, dtype=float)
        else:
            features_example = np.array(args.features, dtype=float)
            if features_example.shape[0] != n_features:
                print(
                    f"[ERREUR] --features contient {features_example.shape[0]} valeurs, "
                    f"mais le modèle attend {n_features} features.",
                    file=sys.stderr,
                )
                sys.exit(2)
    else:
        # Modèles linéaires/logistiques
        coef = np.asarray(est.coef_, dtype=float).reshape(-1)
        intercept = float(np.asarray(est.intercept_, dtype=float).reshape(()))

        if args.features is None:
            features_example = np.ones_like(coef, dtype=float)
        else:
            features_example = np.array(args.features, dtype=float)
            if features_example.shape[0] != coef.shape[0]:
                print(
                    f"[ERREUR] --features contient {features_example.shape[0]} valeurs, "
                    f"mais le modèle attend {coef.shape[0]} features.",
                    file=sys.stderr,
                )
                sys.exit(2)

    # 4) Générer le code C
    if model_type == 'tree':
        c_src = generate_tree_c_source(est, features_example, os.path.basename(sys.argv[0]), n_features)
    else:
        c_src = generate_c_source(intercept, coef, features_example, os.path.basename(sys.argv[0]), model_type)

    # 5) Sauvegarder
    with open(args.out_c, "w", encoding="utf-8") as f:
        f.write(c_src)
    print(f"[OK] Code C généré -> {args.out_c}")

    # Afficher la commande de compilation
    out_bin = os.path.splitext(args.out_c)[0]
    compile_cmd = ["gcc", "-O3", "-std=c11", args.out_c, "-o", out_bin]
    print("[INFO] Commande de compilation :")
    print("      " + " ".join(compile_cmd))

    # 6) Compiler (optionnel) et exécuter (optionnel)
    if args.compile or args.run or args.compare:
        try:
            subprocess.check_call(compile_cmd)
            print(f"[OK] Binaire compilé -> {out_bin}")
        except FileNotFoundError:
            print("[ERREUR] gcc introuvable. Installez Xcode command line tools (macOS) ou gcc.", file=sys.stderr)
            sys.exit(3)
        except subprocess.CalledProcessError:
            print("[ERREUR] Échec de compilation.", file=sys.stderr)
            sys.exit(4)

    if args.run:
        print("[INFO] Exécution du binaire …")
        subprocess.check_call([f"./{out_bin}"])

    # 7) Comparaison avec model.predict (optionnel)
    if args.compare:
        if model_type == 'tree':
            # Prédiction avec l'arbre de décision
            y_py = float(est.predict([features_example])[0])
            print(f"[PY]  predict(x) = {y_py:.6f}")
        else:
            # Calculer la prédiction Python pour les modèles linéaires/logistiques
            z = float(intercept + np.dot(coef, features_example))
            if model_type == 'logistic':
                # Appliquer sigmoid pour régression logistique
                y_py = 1.0 / (1.0 + np.exp(-z))
                print(f"[PY]  predict_proba(x) = {y_py:.6f}")
            else:
                y_py = z
                print(f"[PY]  predict(x) = {y_py:.6f}")

        # On exécute le binaire et on lit y_hat
        proc = subprocess.run([f"./{out_bin}"], capture_output=True, text=True)
        if proc.returncode != 0:
            print("[ERREUR] Échec d'exécution du binaire pour la comparaison.", file=sys.stderr)
            sys.exit(5)
        # sortie attendue: "y_hat=..."
        line = proc.stdout.strip().splitlines()[-1]
        if "y_hat=" in line:
            y_c = float(line.split("y_hat=", 1)[1])
            print(f"[C ]  y_hat     = {y_c:.6f}")
            print(f"[Δ ]  abs diff  = {abs(y_py - y_c):.6e}")
        else:
            print("[AVERTISSEMENT] Sortie inattendue du binaire :", line)


if __name__ == "__main__":
    main()
