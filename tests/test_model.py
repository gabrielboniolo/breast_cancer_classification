import pickle
import pytest
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score

ML_DIR = Path(__file__).parent.parent / "ml"

MODEL_PATH  = ML_DIR / "melhor_modelo.pkl"
SCALER_PATH = ML_DIR / "scaler.pkl"
COLUMNS_PATH = ML_DIR / "colunas.pkl"

SEED = 42
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
]

MIN_ACCURACY  = 0.90
MIN_RECALL    = 0.90 
MIN_F1        = 0.88

@pytest.fixture(scope="module")
def artefacts():
    """Load the three pkl files once for the entire module."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    columns = None
    if COLUMNS_PATH.exists():
        with open(COLUMNS_PATH, "rb") as f:
            columns = pickle.load(f)
    return model, scaler, columns


@pytest.fixture(scope="module")
def test_data():
    dataset = load_breast_cancer(as_frame=True)
    features = list(dataset.feature_names[:10])
    X = dataset.data[features]
    y = dataset.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_test, y_test


@pytest.fixture(scope="module")
def predictions(artefacts, test_data):
    model, scaler, _ = artefacts
    X_test, y_test = test_data
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    return y_test, y_pred

class TestArtefacts:

    def test_model_file_exists(self):
        assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"

    def test_scaler_file_exists(self):
        assert SCALER_PATH.exists(), f"Scaler not found at {SCALER_PATH}"

    def test_columns_file_exists(self):
        assert COLUMNS_PATH.exists(), f"Columns file not found at {COLUMNS_PATH}"

    def test_model_loads_without_error(self, artefacts):
        model, _, _ = artefacts
        assert model is not None

    def test_scaler_loads_without_error(self, artefacts):
        _, scaler, _ = artefacts
        assert scaler is not None

    def test_columns_is_a_list(self, artefacts):
        _, _, columns = artefacts
        assert isinstance(columns, list)

    def test_columns_has_10_features(self, artefacts):
        _, _, columns = artefacts
        assert len(columns) == 10

    def test_model_has_predict_method(self, artefacts):
        model, _, _ = artefacts
        assert hasattr(model, "predict")

    def test_model_has_predict_proba_method(self, artefacts):
        model, _, _ = artefacts
        assert hasattr(model, "predict_proba")

    def test_scaler_has_transform_method(self, artefacts):
        _, scaler, _ = artefacts
        assert hasattr(scaler, "transform")

class TestScaler:

    def test_scaler_output_shape(self, artefacts, test_data):
        _, scaler, _ = artefacts
        X_test, _ = test_data
        X_scaled = scaler.transform(X_test)
        assert X_scaled.shape == X_test.shape

    def test_scaler_produces_finite_values(self, artefacts, test_data):
        _, scaler, _ = artefacts
        X_test, _ = test_data
        X_scaled = scaler.transform(X_test)
        assert np.all(np.isfinite(X_scaled))

    def test_scaler_mean_close_to_zero(self, artefacts, test_data):
        """StandardScaler trained on train set — test set mean won't be exactly 0
        but should be reasonably close."""
        _, scaler, _ = artefacts
        X_test, _ = test_data
        X_scaled = scaler.transform(X_test)
        assert abs(X_scaled.mean()) < 1.0

    def test_single_sample_scaling(self, artefacts):
        _, scaler, _ = artefacts
        sample = np.array([[14.12, 19.29, 91.97, 654.8,
                            0.096, 0.104, 0.088, 0.048, 0.181, 0.062]])
        scaled = scaler.transform(sample)
        assert scaled.shape == (1, 10)
        assert np.all(np.isfinite(scaled))

class TestPredictionOutput:

    def test_predict_output_shape(self, artefacts, test_data):
        model, scaler, _ = artefacts
        X_test, y_test = test_data
        y_pred = model.predict(scaler.transform(X_test))
        assert y_pred.shape == y_test.shape

    def test_predict_returns_only_valid_labels(self, artefacts, test_data):
        model, scaler, _ = artefacts
        X_test, _ = test_data
        y_pred = model.predict(scaler.transform(X_test))
        assert set(y_pred).issubset({0, 1})

    def test_predict_proba_output_shape(self, artefacts, test_data):
        model, scaler, _ = artefacts
        X_test, _ = test_data
        proba = model.predict_proba(scaler.transform(X_test))
        assert proba.shape == (len(X_test), 2)

    def test_predict_proba_rows_sum_to_one(self, artefacts, test_data):
        model, scaler, _ = artefacts
        X_test, _ = test_data
        proba = model.predict_proba(scaler.transform(X_test))
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_values_between_0_and_1(self, artefacts, test_data):
        model, scaler, _ = artefacts
        X_test, _ = test_data
        proba = model.predict_proba(scaler.transform(X_test))
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_single_sample_prediction(self, artefacts):
        model, scaler, _ = artefacts
        sample = np.array([[14.12, 19.29, 91.97, 654.8,
                            0.096, 0.104, 0.088, 0.048, 0.181, 0.062]])
        scaled = scaler.transform(sample)
        pred = model.predict(scaled)
        assert pred[0] in (0, 1)

    def test_single_sample_proba(self, artefacts):
        model, scaler, _ = artefacts
        sample = np.array([[14.12, 19.29, 91.97, 654.8,
                            0.096, 0.104, 0.088, 0.048, 0.181, 0.062]])
        scaled = scaler.transform(sample)
        proba = model.predict_proba(scaled)
        assert abs(proba[0].sum() - 1.0) < 1e-6

class TestModelPerformance:

    def test_accuracy_above_threshold(self, predictions):
        y_test, y_pred = predictions
        acc = accuracy_score(y_test, y_pred)
        assert acc >= MIN_ACCURACY, f"Accuracy {acc:.4f} below minimum {MIN_ACCURACY}"

    def test_malignant_recall_above_threshold(self, predictions):
        y_test, y_pred = predictions
        rec = recall_score(y_test, y_pred, pos_label=0)
        assert rec >= MIN_RECALL, f"Malignant recall {rec:.4f} below minimum {MIN_RECALL}"

    def test_f1_score_above_threshold(self, predictions):
        y_test, y_pred = predictions
        f1 = f1_score(y_test, y_pred, pos_label=0)
        assert f1 >= MIN_F1, f"F1-Score {f1:.4f} below minimum {MIN_F1}"

    def test_both_classes_are_predicted(self, predictions):
        _, y_pred = predictions
        assert 0 in y_pred, "Model never predicted Malignant"
        assert 1 in y_pred, "Model never predicted Benign"

    def test_model_is_deterministic(self, artefacts, test_data):
        model, scaler, _ = artefacts
        X_test, _ = test_data
        X_scaled = scaler.transform(X_test)
        pred_1 = model.predict(X_scaled)
        pred_2 = model.predict(X_scaled)
        assert np.array_equal(pred_1, pred_2)
