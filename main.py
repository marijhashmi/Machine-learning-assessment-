from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io

app = FastAPI()

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:63342"],  # Replace with ["http://127.0.0.1:5500"] if using Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        print("\n[INFO] Received CSV columns:", df.columns.tolist())
        print("[INFO] First few rows:\n", df.head())

        # Drop irrelevant columns
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # Encode categorical features
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0}).fillna(-1)
        df['Embarked'] = df['Embarked'].map({'S': 2, 'C': 0, 'Q': 1}).fillna(-1)

        print("[INFO] After mapping:\n", df.head())

        # Scale numerical features
        df[['Age', 'Fare']] = scaler.transform(df[['Age', 'Fare']])

        print("[INFO] After scaling:\n", df.head())

        # Predict
        preds = model.predict(df)
        print("[INFO] Predictions:", preds.tolist())

        # Map predictions to text
        result = ["Survived" if pred == 1 else "Not Survived" for pred in preds]
        return {"predictions": result}


    except Exception as e:
        print("[ERROR]", str(e))
        return JSONResponse(status_code=400, content={"error": str(e)})
