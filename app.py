from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from sklearn.metrics import accuracy_score, mean_squared_error

from core.inference import predict_project_type
from core.evaluator import generate_evaluation
from core.report_generator import generate_report

# ---------------- Evaluation Mode ----------------
def detect_evaluation_mode(project_type: str) -> str:
    predictive = [
        "ML_REGRESSION",
        "ML_CLASSIFICATION",
        "DL_IMAGE_CLASSIFICATION",
        "TIME_SERIES",
        "REINFORCEMENT_LEARNING"
    ]
    engineering = ["WEB_APPLICATION", "BUSINESS_ANALYTICS"]

    if project_type in predictive:
        return "PREDICTIVE_PERFORMANCE"
    if project_type in engineering:
        return "ENGINEERING_PERFORMANCE"
    return "IMPACT_EVALUATION"


# ---------------- Project Comparator ----------------
def compare_projects(current: dict, previous: dict) -> list:
    insights = []

    if current.get("dataset_size", 0) > previous.get("dataset_size", 0):
        insights.append("Dataset size increased, improving generalization.")

    if current.get("epochs", 0) > previous.get("epochs", 0):
        insights.append("Training epochs increased, allowing deeper learning.")

    if current.get("model") != previous.get("model"):
        insights.append(
            f"Model changed from {previous.get('model')} to {current.get('model')}."
        )

    if not insights:
        insights.append("No major changes compared to previous work.")

    return insights


# ---------------- FastAPI App ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Evaluation Agent is running"}


@app.post("/analyze")
async def analyze_code(
    code: str = Form(...),
    dataset: UploadFile = File(None),
    current_meta: str = Form(None),
    previous_meta: str = Form(None)
):
    project_type = predict_project_type(code)
    evaluation_mode = detect_evaluation_mode(project_type)

    evaluation = generate_evaluation(project_type)
    report = generate_report(project_type, evaluation)

    accuracy = None
    graph_data = {}
    comparison_insights = []

    # ---------- DATASET-BASED EVALUATION ----------
    if dataset is not None:
        df = pd.read_csv(dataset.file)

        # Classification
        if project_type == "ML_CLASSIFICATION" and {"y_true","y_pred"}.issubset(df.columns):
            accuracy = round(accuracy_score(df["y_true"], df["y_pred"]) * 100, 2)
            graph_data = {
                "type": "classification",
                "actual": df["y_true"].tolist(),
                "predicted": df["y_pred"].tolist()
            }

        # Regression
        if project_type == "ML_REGRESSION" and {"y_true","y_pred"}.issubset(df.columns):
            rmse = mean_squared_error(df["y_true"], df["y_pred"], squared=False)
            accuracy = round((1 / (1 + rmse)) * 100, 2)
            graph_data = {
                "type": "regression",
                "actual": df["y_true"].tolist(),
                "predicted": df["y_pred"].tolist()
            }

    # ---------- PROJECT COMPARISON ----------
    if current_meta and previous_meta:
        current = json.loads(current_meta)
        previous = json.loads(previous_meta)
        comparison_insights = compare_projects(current, previous)

    return {
        "project_type": project_type,
        "evaluation_mode": evaluation_mode,
        "accuracy_percentage": accuracy,
        "recommended_metrics": report["recommended_metrics"],
        "improvement_insights": report["improvement_insights"],
        "comparison_insights": comparison_insights,
        "graph_data": graph_data,
        "summary": report["summary"]
    }
