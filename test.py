import pandas as pd
from joblib import load

# Load the model
final_model = load("final_model.joblib")

# Load X_train
X_train = load("X_train.joblib")

# Create new data for prediction
new_data = pd.DataFrame(
    {
        "Financial_Planning": [4],
        "Financial_Goals": [2],
        "Analyze_Financial_Position": [2],
        "Tax_Saving_Schemes": [4],
        "Prompt_Installment_Payments": [4],
        "Profile_Risk": [4],
    }
)

# Prepare new data for prediction
new_data = new_data[X_train.columns]

# Make predictions
predictions = final_model.predict(new_data)
print(f"Predictions for the new data: {predictions}")
