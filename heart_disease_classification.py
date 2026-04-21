from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from mixed_naive_bayes import MixedNB

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

RANDOM_STATE = 42
DATA_PATH = Path("Heart Disease (4).csv")
OUT_DIR = Path("outputs")
PLOTS_DIR = OUT_DIR / "plots"
TREES_DIR = OUT_DIR / "trees"


NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def make_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    TREES_DIR.mkdir(exist_ok=True)


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Remove eventual unnamed index column from CSV export.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Replace textual missing markers by true NaN values.
    df = df.replace("?", np.nan)

    # Force numeric types where expected.
    for col in NUMERIC_COLS + CATEGORICAL_COLS + ["num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary target: 1 if disease is present, else 0.
    df["disease"] = (df["num"] > 0).astype(int)
    return df


def basic_pandas_study(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("1) BASIC DATASET INFO")
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum().sort_values(ascending=False))
    print("\nDescribe numeric:")
    print(df.describe(include=[np.number]).T)


def exploratory_study(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("2) EXPLORATORY STUDY")
    print("=" * 80)

    print("\nTarget distribution (disease):")
    print(df["disease"].value_counts(normalize=True).rename("proportion"))

    # Univariate plots.
    for col in NUMERIC_COLS:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"univariate_numeric_{col}.png", dpi=130)
        plt.close()

    for col in CATEGORICAL_COLS:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[col])
        plt.title(f"Counts of {col}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"univariate_categorical_{col}.png", dpi=130)
        plt.close()

    # Bivariate study with target.
    print("\nNumeric variables grouped by disease:")
    grouped = df.groupby("disease")[NUMERIC_COLS].mean().T
    print(grouped)

    for col in NUMERIC_COLS:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="disease", y=col, data=df)
        plt.title(f"{col} vs disease")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"bivariate_numeric_{col}_vs_disease.png", dpi=130)
        plt.close()

    for col in CATEGORICAL_COLS:
        ct = pd.crosstab(df[col], df["disease"], normalize="index")
        print(f"\nCross-tab normalized for {col}:")
        print(ct)

        plt.figure(figsize=(7, 4))
        sns.countplot(x=col, hue="disease", data=df)
        plt.title(f"{col} vs disease")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"bivariate_categorical_{col}_vs_disease.png", dpi=130)
        plt.close()


def build_base_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_COLS),
            ("cat", cat_pipe, CATEGORICAL_COLS),
        ]
    )


def get_scores(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["avg_precision"] = average_precision_score(y_true, y_score)
    return metrics


def print_scores_block(name: str, scores_train: Dict[str, float], scores_test: Dict[str, float]) -> None:
    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)
    print("Train metrics:", {k: round(v, 4) for k, v in scores_train.items()})
    print("Test metrics :", {k: round(v, 4) for k, v in scores_test.items()})


def get_decision_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        vals = model.decision_function(X)
        vals = np.asarray(vals)
        if vals.ndim == 1:
            return vals
    return None


def plot_roc_pr(name: str, y_true: pd.Series, y_score: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"roc_{name}.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"pr_{name}.png", dpi=130)
    plt.close()


def evaluate_pipeline(name: str, model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, print_reports: bool = False) -> Dict[str, Dict[str, float]]:
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    y_score_train = get_decision_scores(model, X_train)
    y_score_test = get_decision_scores(model, X_test)

    scores_train = get_scores(y_train, y_pred_train, y_score_train)
    scores_test = get_scores(y_test, y_pred_test, y_score_test)

    print_scores_block(name, scores_train, scores_test)

    if print_reports:
        print("\nClassification report (test):")
        print(classification_report(y_test, y_pred_test, digits=4))
        print("Confusion matrix (test):")
        print(confusion_matrix(y_test, y_pred_test))

    if y_score_test is not None:
        plot_roc_pr(name, y_test, y_score_test)

    return {"train": scores_train, "test": scores_test}


def section_naive_bayes(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    print("\n" + "=" * 80)
    print("3) NAIVE BAYES")
    print("=" * 80)

    # Gaussian NB on quantitative variables only.
    pre_num = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_COLS,
            )
        ],
        remainder="drop",
    )
    gnb_pipe = Pipeline([("prep", pre_num), ("model", GaussianNB())])
    evaluate_pipeline("gaussian_nb_numeric_only", gnb_pipe, X_train, y_train, X_test, y_test, print_reports=True)

    # Categorical NB on qualitative variables only.
    pre_cat_ordinal = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    ),
                ]),
                CATEGORICAL_COLS,
            )
        ],
        remainder="drop",
    )

    class SafeCategoricalNB(CategoricalNB):
        def _check_X(self, X):  # type: ignore[override]
            X = np.asarray(X)
            X = np.where(X < 0, 0, X)
            return super()._check_X(X)

    cnb_pipe = Pipeline([("prep", pre_cat_ordinal), ("model", SafeCategoricalNB())])
    evaluate_pipeline("categorical_nb_qualitative_only", cnb_pipe, X_train, y_train, X_test, y_test, print_reports=True)

    # Mixed NB from complementary package for mixed features.
    mixed_prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_COLS,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder()),
                ]),
                CATEGORICAL_COLS,
            ),
        ]
    )

    class MixedNBWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self.model = MixedNB(categorical_features=list(range(len(NUMERIC_COLS), len(NUMERIC_COLS) + len(CATEGORICAL_COLS))))

        def fit(self, X, y):
            self.model.fit(X, y)
            self.is_fitted_ = True
            return self

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict_proba(X)

    mixed_pipe = Pipeline([("prep", mixed_prep), ("model", MixedNBWrapper())])
    evaluate_pipeline("mixed_nb_external_module", mixed_pipe, X_train, y_train, X_test, y_test, print_reports=True)


def section_logistic_regression(base_prep: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    print("\n" + "=" * 80)
    print("4) LOGISTIC REGRESSION")
    print("=" * 80)

    lr_pipe = Pipeline(
        [
            ("prep", clone(base_prep)),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )

    grid = {
        "model__C": [0.01, 0.1, 1, 5, 10, 50],
        "model__solver": ["liblinear", "lbfgs"],
        "model__class_weight": [None, "balanced"],
    }

    gs = GridSearchCV(lr_pipe, grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Best CV F1:", round(gs.best_score_, 4))

    best_lr = gs.best_estimator_
    evaluate_pipeline("logistic_regression_best", best_lr, X_train, y_train, X_test, y_test, print_reports=True)


def section_knn(base_prep: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    print("\n" + "=" * 80)
    print("5) K-NN")
    print("=" * 80)

    knn_pipe = Pipeline(
        [
            ("prep", clone(base_prep)),
            ("model", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }
    gs = GridSearchCV(knn_pipe, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)

    print("Best params (GridSearch):", gs.best_params_)
    print("Best CV F1:", round(gs.best_score_, 4))

    best_knn = gs.best_estimator_
    evaluate_pipeline("knn_grid_best", best_knn, X_train, y_train, X_test, y_test, print_reports=True)

    # Imputation optimization directly in grid search.
    prep_with_tunable_impute = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )

    knn_pipe_tunable = Pipeline(
        [
            ("prep", prep_with_tunable_impute),
            ("model", KNeighborsClassifier()),
        ]
    )

    impute_grid = {
        "prep__num__imputer__strategy": ["mean", "median"],
        "prep__cat__imputer__strategy": ["most_frequent", "constant"],
        "prep__cat__imputer__fill_value": [0],
        "model__n_neighbors": [5, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
    }

    gs_impute = GridSearchCV(knn_pipe_tunable, param_grid=impute_grid, cv=5, scoring="f1", n_jobs=-1)
    gs_impute.fit(X_train, y_train)

    print("Best params with imputation optimization:", gs_impute.best_params_)
    print("Best CV F1:", round(gs_impute.best_score_, 4))

    evaluate_pipeline("knn_imputation_grid_best", gs_impute.best_estimator_, X_train, y_train, X_test, y_test, print_reports=True)

    # Randomized search version.
    rs_dist = {
        "model__n_neighbors": np.arange(3, 31),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }
    rs = RandomizedSearchCV(
        knn_pipe,
        param_distributions=rs_dist,
        n_iter=25,
        scoring="f1",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rs.fit(X_train, y_train)

    print("Best params (RandomizedSearch):", rs.best_params_)
    print("Best CV F1:", round(rs.best_score_, 4))

    evaluate_pipeline("knn_randomized_best", rs.best_estimator_, X_train, y_train, X_test, y_test, print_reports=True)


def section_svm(base_prep: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    print("\n" + "=" * 80)
    print("6) SVM")
    print("=" * 80)

    # Linear SVM study.
    svm_linear_pipe = Pipeline(
        [
            ("prep", clone(base_prep)),
            ("model", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
        ]
    )
    linear_grid = {
        "model__C": [0.01, 0.1, 1, 10, 50],
        "model__class_weight": [None, "balanced"],
    }

    gs_linear = GridSearchCV(svm_linear_pipe, linear_grid, cv=5, scoring="f1", n_jobs=-1)
    gs_linear.fit(X_train, y_train)

    print("Best params linear SVM:", gs_linear.best_params_)
    print("Best CV F1:", round(gs_linear.best_score_, 4))

    evaluate_pipeline("svm_linear_best", gs_linear.best_estimator_, X_train, y_train, X_test, y_test, print_reports=True)

    # Kernel selection with grid search.
    svm_kernel_pipe = Pipeline(
        [
            ("prep", clone(base_prep)),
            ("model", SVC(probability=True, random_state=RANDOM_STATE)),
        ]
    )

    kernel_grid = [
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1, 10, 50],
            "model__gamma": ["scale", 0.01, 0.1, 1],
        },
        {
            "model__kernel": ["poly"],
            "model__C": [0.1, 1, 10],
            "model__degree": [2, 3, 4],
            "model__gamma": ["scale", 0.01, 0.1],
        },
        {
            "model__kernel": ["sigmoid"],
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.01, 0.1],
        },
    ]

    gs_kernel = GridSearchCV(svm_kernel_pipe, kernel_grid, cv=5, scoring="f1", n_jobs=-1)
    gs_kernel.fit(X_train, y_train)

    print("Best params kernel SVM:", gs_kernel.best_params_)
    print("Best CV F1:", round(gs_kernel.best_score_, 4))

    evaluate_pipeline("svm_kernel_best", gs_kernel.best_estimator_, X_train, y_train, X_test, y_test, print_reports=True)


def section_lda(base_prep: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    print("\n" + "=" * 80)
    print("7) LDA")
    print("=" * 80)

    lda_pipe = Pipeline(
        [
            ("prep", clone(base_prep)),
            ("model", LinearDiscriminantAnalysis()),
        ]
    )

    # LDA has few tunable params useful in this context.
    grid = {
        "model__solver": ["svd", "lsqr", "eigen"],
        "model__shrinkage": [None, "auto"],
    }

    # shrinkage only works with lsqr/eigen, so we test each case with conditional grids.
    lda_grid = [
        {"model__solver": ["svd"]},
        {"model__solver": ["lsqr", "eigen"], "model__shrinkage": [None, "auto"]},
    ]

    gs = GridSearchCV(lda_pipe, param_grid=lda_grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)

    print("Best params LDA:", gs.best_params_)
    print("Best CV F1:", round(gs.best_score_, 4))

    evaluate_pipeline("lda_best", gs.best_estimator_, X_train, y_train, X_test, y_test, print_reports=True)


def section_decision_tree(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    print("\n" + "=" * 80)
    print("8) DECISION TREE + GRAPHVIZ")
    print("=" * 80)

    # For trees, we keep ordinal encoded categorical variables.
    tree_prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_COLS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )

    tree_pipe = Pipeline(
        [
            ("prep", tree_prep),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [None, 3, 4, 5, 6, 8, 10],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__ccp_alpha": [0.0, 0.001, 0.005, 0.01],
    }

    gs = GridSearchCV(tree_pipe, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)

    print("Best params decision tree:", gs.best_params_)
    print("Best CV F1:", round(gs.best_score_, 4))

    best_tree = gs.best_estimator_
    evaluate_pipeline("decision_tree_best", best_tree, X_train, y_train, X_test, y_test, print_reports=True)

    # Additional cross-validation report for complete study.
    cv_scores = cross_validate(
        best_tree,
        X_train,
        y_train,
        cv=5,
        scoring={"acc": "accuracy", "prec": "precision", "rec": "recall", "f1": "f1", "auc": "roc_auc"},
        n_jobs=-1,
    )
    print("\nCross-validation means (decision tree):")
    print({k: round(float(np.mean(v)), 4) for k, v in cv_scores.items() if k.startswith("test_")})

    # Build Graphviz representation.
    prep = best_tree.named_steps["prep"]
    model = best_tree.named_steps["model"]
    feature_names = prep.get_feature_names_out()

    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=["no_disease", "disease"],
        filled=True,
        rounded=True,
        special_characters=True,
    )

    dot_path = TREES_DIR / "decision_tree_best.dot"
    dot_path.write_text(dot_data, encoding="utf-8")
    print(f"Graphviz DOT exported to: {dot_path}")

    # Try to render PNG; if Graphviz executable is missing, keep DOT only.
    try:
        graph = graphviz.Source(dot_data)
        render_path = graph.render(filename="decision_tree_best", directory=str(TREES_DIR), format="png", cleanup=True)
        print(f"Graph rendered to: {render_path}")
    except Exception as exc:
        print("Graphviz rendering failed (DOT file still available).")
        print("Reason:", exc)
        print("Install Graphviz system binaries to enable PNG rendering.")



def main() -> None:
    make_dirs()
    df = load_and_prepare(DATA_PATH)

    basic_pandas_study(df)
    exploratory_study(df)

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\n" + "=" * 80)
    print("Train/Test split")
    print("=" * 80)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

    base_prep = build_base_preprocessor()

    section_naive_bayes(X_train, X_test, y_train, y_test)
    section_logistic_regression(base_prep, X_train, X_test, y_train, y_test)
    section_knn(base_prep, X_train, X_test, y_train, y_test)
    section_svm(base_prep, X_train, X_test, y_train, y_test)
    section_lda(base_prep, X_train, X_test, y_train, y_test)
    section_decision_tree(X_train, X_test, y_train, y_test)

    print("\nAll done. Check outputs/ for plots and tree artifacts.")


if __name__ == "__main__":
    main()
