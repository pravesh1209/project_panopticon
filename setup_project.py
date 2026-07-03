#!/usr/bin/env python3
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STRUCTURE = {
    "config": ["config.yaml", "logging.yaml"],
    "src": {
        "__init__.py": "",
        "data": ["__init__.py", "loader.py", "cleaner.py", "merger.py", "features.py", "pipeline.py", "synthetic_generator.py"],
        "cv": ["__init__.py", "gaze_estimator.py", "head_pose.py", "object_detector.py", "face_tracker.py"],
        "audio": ["__init__.py", "transcriber.py", "keyword_detector.py", "noise_analyzer.py"],
        "rules": ["__init__.py", "risk_engine.py"],
        "ml": ["__init__.py", "train.py", "evaluate.py", "threshold_optimizer.py", "explainer.py"],
        "dashboard": {
            "__init__.py": "",
            "app.py": "",
            "pages": ["dashboard.py", "monitoring.py", "timeline.py", "risk_analysis.py", "model_metrics.py", "threshold_analysis.py", "shap_analysis.py", "violations.py"],
            "utils.py": "",
        },
        "utils": ["__init__.py", "logger.py", "metrics.py", "helpers.py"],
    },
    "data": ["raw", "processed", "synthetic"],
    "models": [],
    "notebooks": ["01_data_exploration.ipynb", "02_feature_engineering.ipynb", "03_model_training.ipynb", "executive_summary.ipynb"],
    "dashboard": {"static": [], "templates": []},
    "reports": [],
    "logs": [],
    "tests": ["test_data.py", "test_cv.py", "test_audio.py", "test_rules.py", "test_ml.py", "conftest.py"],
    "assets": [],
}

def create_structure(base, structure):
    for name, content in structure.items():
        path = base / name
        if isinstance(content, list):
            path.mkdir(parents=True, exist_ok=True)
            for item in content:
                item_path = path / item
                if '.' in item:
                    item_path.touch(exist_ok=True)
                else:
                    item_path.mkdir(parents=True, exist_ok=True)
        elif isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

if __name__ == "__main__":
    create_structure(PROJECT_ROOT, STRUCTURE)
    req_path = PROJECT_ROOT / "requirements.txt"
    req_path.write_text("""pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.0.0
opencv-python>=4.6.0
mediapipe>=0.8.0
ultralytics>=8.0.0
openai-whisper>=20231117
librosa>=0.10.0
shap>=0.41.0
matplotlib>=3.5.0
plotly>=5.9.0
seaborn>=0.11.0
streamlit>=1.20.0
joblib>=1.2.0
pyyaml>=6.0
python-dotenv>=0.20.0
pytest>=7.0.0
""")
    print("✅ Project structure created successfully.")
