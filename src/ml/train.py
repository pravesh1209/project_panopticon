"""
Model Training Module
Trains multiple classifiers with built‑in class imbalance handling.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

def get_features_target(df, target_col='is_cheating', exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    drop_cols = [target_col] + exclude_cols
    drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]
    return X, y

def train_models(X_train, y_train, random_state=42, use_gpu=False):
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            tree_method='gpu_hist' if use_gpu else 'auto'
        ),
        'LightGBM': LGBMClassifier(
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            device='gpu' if use_gpu else 'cpu'
        ),
        'CatBoost': CatBoostClassifier(
            auto_class_weights='Balanced',
            random_seed=random_state,
            verbose=0,
            n_estimators=50,
            depth=6,
            learning_rate=0.1,
            task_type='GPU' if use_gpu else 'CPU'
        )
    }

    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def save_models(models, path='models/'):
    os.makedirs(path, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(path, f"{name}.pkl"))
    print(f"Models saved to {path}")

def load_models(path='models/'):
    models = {}
    for file in os.listdir(path):
        if file.endswith('.pkl'):
            name = file.replace('.pkl', '')
            models[name] = joblib.load(os.path.join(path, file))
    return models