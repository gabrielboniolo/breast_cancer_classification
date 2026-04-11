import pickle
from pathlib import Path

ML_DIR = Path("ml")

def load_classifier():
    
    classifier_path = ML_DIR / "melhor_modelo.pkl"
    scaler_path = ML_DIR / "scaler.pkl"
    columns_path = ML_DIR / "colunas.pkl"

    if not classifier_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Classifier files '.pkl' were not found.")

    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    columns = None
    if columns_path.exists():
        with open(columns_path, "rb") as f:
            columns = pickle.load(f)

    return classifier, scaler, columns