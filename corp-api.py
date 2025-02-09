from fastapi import FastAPI
import torch
import numpy as np
import pickle
from pydantic import BaseModel

# Load the trained model
model = torch.load("crop_prediction_model.pth", map_location=torch.device('cpu'))
model.eval()

# Load the scaler and label encoder
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

app = FastAPI()

class CropPredictionRequest(BaseModel):
    temperature: float
    rainfall: float
    humidity: float
    soil_pH: float
    soil_moisture: float

@app.post("/predict")
def predict_crop(data: CropPredictionRequest):
    input_data = np.array([[data.temperature, data.rainfall, data.humidity, data.soil_pH, data.soil_moisture]])
    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, axis=1).item()

    predicted_crop = label_encoder.inverse_transform([predicted_class])[0]
    return {"predicted_crop": predicted_crop}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
