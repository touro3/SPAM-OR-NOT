from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from src.preprocessing import preprocess_features

app = FastAPI()

# Carrega o modelo ensemble treinado
model = joblib.load("voting_clf.pkl")

# Define o esquema de entrada
class InputData(BaseModel):
    features: list[float]  # 57 valores do Spambase (48 palavras especifica, 6 caracteres especiais, 2 características de comprimento, 1 total de letras maiusculas)
@app.get("/")
def read_root():
    return {"message": "API funcionando!"}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Prepara e transforma os dados de entrada
        X = np.array(input_data.features).reshape(1, -1)
        X = preprocess_features(X)

        # Realiza a predição
        prediction = model.predict(X)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
