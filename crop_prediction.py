import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic dataset
num_samples = 500
temperature = np.random.uniform(15, 40, num_samples)  # in Celsius
rainfall = np.random.uniform(50, 300, num_samples)  # in mm
humidity = np.random.uniform(20, 90, num_samples)  # in percentage
soil_pH = np.random.uniform(4.5, 8.5, num_samples)  # pH value
soil_moisture = np.random.uniform(10, 40, num_samples)  # in percentage
crop_labels = ["Wheat", "Rice", "Maize", "Soybean", "Barley", "Cotton", "Sugarcane", "Potato", "Tomato", "Sunflower"]
crop_conditions = np.random.choice(crop_labels, num_samples)

# Create DataFrame
df = pd.DataFrame({
    "Temperature": temperature,
    "Rainfall": rainfall,
    "Humidity": humidity,
    "Soil_pH": soil_pH,
    "Soil_Moisture": soil_moisture,
    "Crop": crop_conditions
})

# Encode crop labels
label_encoder = LabelEncoder()
df["Crop"] = label_encoder.fit_transform(df["Crop"])

# Define features and target
X = df.drop(columns=["Crop"]).values
y = df["Crop"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the Neural Network Model
class CropPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CropPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Define model parameters
input_size = X_train.shape[1]
hidden_size = 16
output_size = len(label_encoder.classes_)

# Initialize model
model = CropPredictionModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate model accuracy
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = torch.argmax(outputs, axis=1)
    accuracy = (predictions == y_test_tensor).float().mean()

print(f"Model Accuracy: {accuracy.item() * 100:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "crop_prediction_model.pth")
print("Model saved as 'crop_prediction_model.pth'")

# Save the scaler and label encoder for future use
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Scaler and Label Encoder saved.")
