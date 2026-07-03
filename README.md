# Project Panopticon – Intelligent Exam Proctoring

## Overview
A production-grade AI system to detect cheating during online exams while minimizing false accusations. Built with time-series analysis, computer vision, audio intelligence, and explainable AI.

## Features
- Asynchronous data merge (Pandas merge_asof)
- Advanced feature engineering (rolling windows, EMA)
- Multiple ML models with class imbalance handling
- Threshold optimization to ensure 90%+ precision
- SHAP explainability
- Streamlit dashboard for monitoring

## Installation
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -e .