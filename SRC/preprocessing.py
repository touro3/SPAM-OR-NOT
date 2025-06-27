from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

def preprocess_data(df):
    """
    Pré-processa o dataset completo para treino/validação.
    """
    X = df.drop('is_spam', axis=1)
    y = df['is_spam']
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Salvar o scaler para uso posterior (ex: no FastAPI)
    joblib.dump(scaler, "scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test


def preprocess_features(X):
    """
    Pré-processa uma amostra individual para predição.
    Espera um array NumPy 2D com shape (1, 57)
    """
    if isinstance(X, list):
        X = np.array(X).reshape(1, -1)

    # Carregar o scaler previamente salvo
    scaler = joblib.load("scaler.pkl")
    X_scaled = scaler.transform(X)
    return X_scaled
