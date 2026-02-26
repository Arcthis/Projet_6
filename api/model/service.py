import bentoml
from bentoml.io import JSON
import pandas as pd
from pydantic import BaseModel

# Charger le modèle BentoML
model_runner = bentoml.sklearn.get("linear_regression_model").to_runner()
svc = bentoml.Service("energy_prediction_service", runners=[model_runner])

BUILDING_TYPE_MAPPING = {
    "Residential": 0,
    "NonResidential": 1,
}

FEATURE_ORDER = ["BuildingType", "NumberofFloors", "PropertyGFATotal", "PropertyGFAParking", "TotalGHGEmissions", "Outlier"]

class InputData(BaseModel):
    BuildingType: str
    NumberofFloors: float
    PropertyGFATotal: float
    PropertyGFAParking: float
    TotalGHGEmissions: float

@svc.api(input=JSON(pydantic_model=InputData), output=JSON(), route="/api/v1/energy")
def predict(data: InputData):
    input_data = data.dict()

    input_data['BuildingType'] = BUILDING_TYPE_MAPPING.get(input_data['BuildingType'], 0)
    
    # Ajouter Outlier
    input_data['Outlier'] = 0

    # Convertir en DataFrame + colonnes
    input_data = pd.DataFrame([input_data])
    input_data = input_data[FEATURE_ORDER]

    # Prédiction
    prediction = model_runner.predict.run(input_data)
    
    return {"predicted_energy_use": prediction[0]}

# exemple de requête :

# $headers = @{ "Content-Type" = "application/json" }
# $body = @{
#     BuildingType = "NonResidential"
#     NumberofFloors = 10
#     PropertyGFATotal = 8000
#     PropertyGFAParking = 200
#     TotalGHGEmissions = 50
# } | ConvertTo-Json

# Invoke-RestMethod -Uri "http://127.0.0.1:3000/predict" -Method Post -Headers $headers -Body $body


