import bentoml
import pandas as pd
import pickle
from pydantic import BaseModel
from bentoml.io import JSON
from sklearn.preprocessing import StandardScaler

model_ref = bentoml.sklearn.get("linear_regression_model")
model = model_ref.load_model()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)  # Charger le scaler sauvegardé

# Créer le service BentoML
svc = bentoml.Service("energy_prediction_service", models=[model_ref])

BUILDING_TYPE_MAPPING = {
    "NonResidential": 0,
    "NonResidential COS": 1,
    "NonResidential WA": 2,
}
FEATURE_ORDER = ["BuildingAge", "BuildingType", "NumberofFloors", "PropertyGFATotal", "PropertyGFAParking", "TotalGHGEmissions", "Outlier"]

# Définition du format d'entrée avec Pydantic
class InputData(BaseModel):
    BuildingAge: int
    BuildingType: str
    NumberofFloors: float
    PropertyGFATotal: float
    PropertyGFAParking: float
    TotalGHGEmissions: float

# Définition du endpoint API
@svc.api(input=JSON(pydantic_model=InputData), output=JSON(), route="/api/v1/energy")
def predict(data: InputData):
    input_data = data.dict()
    # Encodage de la variable catégorielle
    input_data['BuildingType'] = BUILDING_TYPE_MAPPING.get(input_data['BuildingType'], 0)
    input_data['Outlier'] = 0
    # Convertir en DataFrame et ordonner les colonnes
    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_ORDER]
    # Standardiser les données (IMPORTANT)
    input_scaled = scaler.transform(input_df)  # Appliquer le scaler
    input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURE_ORDER)  # Convertir en DataFrame
    # Faire la prédiction avec le modèle
    prediction = model.predict(input_scaled_df)

    return {"predicted_energy_use": prediction[0]}

# API Request Example :
# 
# curl -X POST http://localhost:3000/api/v1/energy \
    #  -H "Content-Type: application/json" \
    #  -d '{
    #        "BuildingAge": 90,
    #        "BuildingType": "NonResidential",
    #        "NumberofFloors": 5,
    #        "PropertyGFATotal": 15000,
    #        "PropertyGFAParking": 2000,
    #        "TotalGHGEmissions": 50
    #      }' 