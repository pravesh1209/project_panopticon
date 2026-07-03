"""
Project Panopticon – Complete Streamlit Dashboard
---------------------------------------------------
Displays 200 students with violation tracking, model evaluation,
and threshold analysis with flagged test samples.
All Streamlit deprecation warnings fixed (use_container_width → width='stretch').
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="Project Panopticon", layout="wide")

# ----------------------------------------------------------------------
# CUSTOM DARK THEME CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .main { background: #0e1117; color: #ffffff; }
    h1, h2, h3, h4, h5, h6, p, label, .stSelectbox label {
        color: #ffffff !important;
    }
    .stSelectbox > div > div {
        background: #1e1e2e;
        color: #ffffff;
    }
    .stSelectbox > div > div > div { color: #ffffff; }
    .stDataFrame {
        background: #1e1e2e;
    }
    .stDataFrame table {
        color: #ffffff !important;
    }
    .stDataFrame thead tr th {
        background: #2d2d44 !important;
        color: #ffffff !important;
    }
    .stDataFrame tbody tr td {
        background: #1a1a2e !important;
        color: #ffffff !important;
    }
    .stDataFrame tbody tr:hover td {
        background: #2a2a44 !important;
    }
    .streamlit-expanderHeader {
        color: #ffffff !important;
        background: #1e1e2e !important;
    }
    .streamlit-expanderContent {
        background: #1a1a2e !important;
    }
    .stMetric {
        background: #1e1e2e;
        padding: 10px;
        border-radius: 10px;
    }
    .stMetric label { color: #ffffff !important; }
    .stMetric div { color: #ffffff !important; }
    .info-box {
        background: #1a2744;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1976d2;
        color: #ffffff;
    }
    .warning-box {
        background: #2a1a0a;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #f57c00;
        color: #ffffff;
    }
    .success-box {
        background: #0a2a1a;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4caf50;
        color: #ffffff;
    }
    .danger-box {
        background: #2a0a0a;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #f44336;
        color: #ffffff;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.9em;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Project Panopticon – Intelligent Proctoring")

# ----------------------------------------------------------------------
# GENERATE STUDENT DATA (200 students)
# ----------------------------------------------------------------------
@st.cache_data
def get_students():
    np.random.seed(42)
    n = 200

    cheat_types = ['Tab Switch', 'Copy-Paste', 'Screenshot', 'Camera Off', 'Background Noise']
    violation_probs = [0.30, 0.20, 0.15, 0.20, 0.15]
    violation_counts = [0, 1, 2, 3, 4]  # 4 means 4 or more

    students_list = []
    for i in range(1, n + 1):
        r = np.random.random()
        cum_prob = 0
        for vc, prob in zip(violation_counts, violation_probs):
            cum_prob += prob
            if r <= cum_prob:
                violation_count = vc
                break
        else:
            violation_count = 0

        if violation_count == 4:
            violation_count = np.random.randint(4, 7)

        violations = []
        for _ in range(violation_count):
            violations.append(np.random.choice(cheat_types))

        if violation_count > 0:
            counter = Counter(violations)
            most_common = counter.most_common(1)[0][0]
            cheating_type = most_common
            summary_parts = [f"{v} ({cnt}x)" for v, cnt in counter.items()]
            violation_summary = ", ".join(summary_parts)
        else:
            cheating_type = 'None'
            violation_summary = ''

        if violation_count == 0:
            risk_score = np.random.uniform(0, 10)
            is_cheating = 0
            risk_category = 'Low'
        elif violation_count == 1:
            risk_score = np.random.uniform(5, 20)
            is_cheating = 0
            risk_category = 'Low'
        elif violation_count == 2:
            risk_score = np.random.uniform(15, 30)
            is_cheating = 0
            risk_category = 'Low'
        elif violation_count == 3:
            risk_score = np.random.uniform(30, 50)
            is_cheating = 1
            risk_category = 'Medium'
        else:
            risk_score = np.random.uniform(50, 100)
            is_cheating = 1
            risk_category = 'High'

        risk_score = min(100, max(0, risk_score + np.random.normal(0, 5)))

        students_list.append({
            'student_id': i,
            'name': f'Student_{i:03d}',
            'risk_score': risk_score,
            'is_cheating': is_cheating,
            'cheating_type': cheating_type,
            'risk_category': risk_category,
            'violation_count': violation_count,
            'violation_summary': violation_summary
        })

    students = pd.DataFrame(students_list)
    students.loc[students['risk_category'].isin(['Medium', 'High']), 'is_cheating'] = 1
    mask = (students['risk_category'].isin(['Medium', 'High'])) & (students['cheating_type'] == 'None')
    if mask.any():
        students.loc[mask, 'cheating_type'] = np.random.choice(cheat_types, size=mask.sum())
        students.loc[mask, 'violation_summary'] = 'Multiple violations'

    return students

students_df = get_students()

# ----------------------------------------------------------------------
# GENERATE TIMELINE FOR EACH STUDENT
# ----------------------------------------------------------------------
@st.cache_data
def generate_timelines(students_df):
    all_timelines = []
    for _, student in students_df.iterrows():
        np.random.seed(student['student_id'] * 7 + 13)
        timestamps = pd.date_range(start='2026-06-26 09:25:00', periods=60, freq='5s')
        base = student['risk_score'] * 0.4
        if student['is_cheating']:
            risk = base + np.cumsum(np.random.randn(60) * 0.8) + np.random.randn(60) * 12
            risk = np.clip(risk, 10, 100)
            for _ in range(3):
                idx = np.random.randint(30, 55)
                risk[idx] += np.random.uniform(15, 35)
            risk = np.clip(risk, 10, 100)
        else:
            risk = base + np.random.randn(60) * 3
            risk = np.clip(risk, 5, 45)
        all_timelines.append(pd.DataFrame({
            'timestamp': timestamps,
            'risk_score': risk,
            'student_id': student['student_id']
        }))
    return pd.concat(all_timelines, ignore_index=True)

timeline_df = generate_timelines(students_df)

# ----------------------------------------------------------------------
# LOAD ACTUAL TELEMETRY DATA FOR MODEL TRAINING
# ----------------------------------------------------------------------
@st.cache_data
def load_telemetry():
    try:
        video = pd.read_csv('data/synthetic/video_telemetry.csv', parse_dates=['timestamp'])
        audio = pd.read_csv('data/synthetic/audio_telemetry.csv', parse_dates=['timestamp'])
        labels = pd.read_csv('data/synthetic/labels.csv', parse_dates=['timestamp'])
        events = pd.read_csv('data/synthetic/system_events.csv', parse_dates=['timestamp'])
        df = pd.merge(video, audio, on='timestamp', how='outer')
        df = pd.merge(df, labels, on='timestamp', how='left')
        events_pivot = events.pivot_table(
            index='timestamp', columns='event_type', values='event_count',
            aggfunc='sum', fill_value=0
        ).reset_index()
        events_pivot.rename(columns={'tab_switch': 'tab_switches', 'copy_paste': 'copy_paste'}, inplace=True)
        for col in ['tab_switches', 'copy_paste']:
            if col not in events_pivot.columns:
                events_pivot[col] = 0
        df = pd.merge_asof(df.sort_values('timestamp'), events_pivot.sort_values('timestamp'), on='timestamp', direction='backward')
        df['tab_switches'] = df['tab_switches'].fillna(0).astype(int)
        df['copy_paste'] = df['copy_paste'].fillna(0).astype(int)
        df['is_cheating'] = df['is_cheating'].fillna(0).astype(int)
        df['eye_gaze_angle'] = df['eye_gaze_angle'].ffill().fillna(0)
        df['audio_db'] = df['audio_db'].interpolate(method='linear', limit_direction='both').fillna(df['audio_db'].median())
        df['gaze_rolling_10s'] = df['eye_gaze_angle'].rolling(10, min_periods=1).mean()
        df['audio_rolling_10s'] = df['audio_db'].rolling(10, min_periods=1).mean()
        df.dropna(subset=['gaze_rolling_10s', 'audio_rolling_10s'], inplace=True)
        return df
    except Exception as e:
        st.warning(f"Could not load telemetry data: {e}")
        return pd.DataFrame()

telemetry_df = load_telemetry()

# ----------------------------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

@st.cache_resource
def train_model(df):
    if df.empty:
        return None, None, None
    features = ['eye_gaze_angle', 'audio_db', 'tab_switches', 'gaze_rolling_10s', 'audio_rolling_10s']
    X = df[features]
    y = df['is_cheating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = None, None, None
if not telemetry_df.empty:
    model, X_test, y_test = train_model(telemetry_df)

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
page = st.sidebar.selectbox(
    "📋 Navigation",
    ["📊 Dashboard", "📹 Monitoring", "📈 Timeline", "⚖️ Risk Analysis",
     "📊 Model Metrics", "🎯 Threshold Analysis", "🚨 Violations"]
)

# ----------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ----------------------------------------------------------------------
if page == "📊 Dashboard":
    st.markdown("### 📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Students", len(students_df))
    col2.metric("⚠️ Cheaters", students_df['is_cheating'].sum())
    col3.metric("✅ Innocent", (students_df['is_cheating'] == 0).sum())
    col4.metric("📊 Avg Risk", f"{students_df['risk_score'].mean():.1f}")

    fig = px.histogram(students_df, x='risk_score', nbins=30,
                       color='risk_category',
                       title='Risk Score by Category',
                       labels={'risk_score': 'Risk Score', 'count': 'Number of Students'},
                       color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, width='stretch')

    st.subheader("🔴 Top 10 Highest Risk Students")
    top = students_df.nlargest(10, 'risk_score')[['student_id', 'name', 'risk_category', 'risk_score', 'cheating_type', 'violation_count', 'violation_summary']]
    top['risk_score'] = top['risk_score'].round(1)
    st.dataframe(top, width='stretch')

# ----------------------------------------------------------------------
# PAGE 2: MONITORING
# ----------------------------------------------------------------------
elif page == "📹 Monitoring":
    st.markdown("### 📹 Live Student Monitoring")
    student_id = st.selectbox("🎯 Select Student ID", students_df['student_id'])
    student = students_df[students_df['student_id'] == student_id].iloc[0]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#2a2a3a')
        circle = plt.Circle((5, 6), 1.5, color='#4a4a8a', ec='white')
        ax.add_patch(circle)
        ax.plot(4.5, 6.5, 'wo', markersize=4)
        ax.plot(5.5, 6.5, 'wo', markersize=4)
        ax.plot(5, 5.5, 'w_', markersize=12)
        ax.text(5, 2, f"Student {student_id}", ha='center', fontsize=14, fontweight='bold', color='white')
        ax.text(5, 1.5, "Live Feed", ha='center', fontsize=10, color='gray')
        if student['is_cheating']:
            ax.text(5, 9, "CHEATING", ha='center', fontsize=20, color='red', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'))
        else:
            ax.text(5, 9, "NORMAL", ha='center', fontsize=20, color='lime', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'))
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("📋 Student Details")
        st.write(f"**Name:** {student['name']}")
        st.write(f"**ID:** {student['student_id']}")
        status = "⚠️ Cheating" if student['is_cheating'] else "✅ Innocent"
        st.write(f"**Status:** {status}")
        st.write(f"**Risk Score:** {student['risk_score']:.1f}/100")
        st.write(f"**Category:** {student['risk_category']}")
        st.write(f"**Cheating Type:** {student['cheating_type']}")
        st.write(f"**Violation Count:** {student['violation_count']}")
        if student['violation_summary']:
            st.write(f"**Violation Summary:** {student['violation_summary']}")
        if student['is_cheating']:
            st.error("🔴 Suspicious activity detected!")
        else:
            st.success("✅ No suspicious activity.")

# ----------------------------------------------------------------------
# PAGE 3: TIMELINE
# ----------------------------------------------------------------------
elif page == "📈 Timeline":
    st.markdown("### 📈 Student Risk Timeline")
    student_id = st.selectbox("🎯 Select Student ID", students_df['student_id'])
    student = students_df[students_df['student_id'] == student_id].iloc[0]
    st_timeline = timeline_df[timeline_df['student_id'] == student_id]

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Student:** {student['name']} (ID: {student_id})")
        st.write(f"**Status:** {'⚠️ Cheating' if student['is_cheating'] else '✅ Innocent'}")
        st.write(f"**Overall Risk:** {student['risk_score']:.1f}")
        st.write(f"**Violations:** {student['violation_count']}")
        if student['violation_summary']:
            st.write(f"**Summary:** {student['violation_summary']}")
    with col2:
        st.write(f"**Peak Risk:** {st_timeline['risk_score'].max():.1f}")
        st.write(f"**Average Risk:** {st_timeline['risk_score'].mean():.1f}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhspan(0, 30, alpha=0.2, color='green', label='Low')
    ax.axhspan(30, 40, alpha=0.2, color='orange', label='Medium')
    ax.axhspan(40, 100, alpha=0.2, color='red', label='High')
    ax.plot(st_timeline['timestamp'], st_timeline['risk_score'], 'cyan', linewidth=2, label='Risk Score')
    ax.axhline(y=40, color='red', linestyle='--', linewidth=2, label='Cheating Threshold (40)')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Risk Score')
    ax.set_facecolor('#1a1a2e')
    ax.legend(loc='upper left')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------------------------------------------------
# PAGE 4: RISK ANALYSIS
# ----------------------------------------------------------------------
elif page == "⚖️ Risk Analysis":
    st.markdown("### ⚖️ Risk Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Low Risk", (students_df['risk_category'] == 'Low').sum())
    col2.metric("🟠 Medium Risk", (students_df['risk_category'] == 'Medium').sum())
    col3.metric("🔴 High Risk", (students_df['risk_category'] == 'High').sum())

    display_df = students_df.copy()
    display_df['status'] = display_df['is_cheating'].map({0: '✅ Innocent', 1: '⚠️ Cheating'})
    display_df['risk_score'] = display_df['risk_score'].round(1)
    st.dataframe(display_df[['student_id', 'name', 'status', 'risk_category', 'risk_score', 'cheating_type', 'violation_count', 'violation_summary']],
                 width='stretch')

# ----------------------------------------------------------------------
# PAGE 5: MODEL METRICS
# ----------------------------------------------------------------------
elif page == "📊 Model Metrics":
    st.markdown("### 📊 Model Performance Metrics")
    if model is None:
        st.warning("Model not trained. Please ensure telemetry data is available.")
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        results = []
        for th in thresholds:
            y_pred = (y_proba >= th).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            results.append({
                'Threshold': th,
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'False Positives': cm[0,1]
            })
        st.dataframe(pd.DataFrame(results), width='stretch')

        fig, ax = plt.subplots(figsize=(10, 6))
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.axhline(y=0.90, color='red', linestyle='--', label='90% Precision Target')
        ax.legend()
        ax.set_title('Precision-Recall Curve')
        st.pyplot(fig)

# ----------------------------------------------------------------------
# PAGE 6: THRESHOLD ANALYSIS (UPDATED)
# ----------------------------------------------------------------------
elif page == "🎯 Threshold Analysis":
    st.markdown("### 🎯 Threshold Optimization")
    if model is None:
        st.warning("Model not trained.")
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_proba)
        valid = precisions[:-1] >= 0.90
        if np.any(valid):
            valid_indices = np.where(valid)[0]
            best_idx = valid_indices[np.argmax(recalls[valid_indices])]
            opt_th = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 0.99
        else:
            opt_th = 0.99
        y_opt = (y_proba >= opt_th).astype(int)
        
        st.success(f"**Optimal Threshold for ≥90% Precision:** {opt_th:.3f}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{precision_score(y_test, y_opt):.3f}")
        col2.metric("Recall", f"{recall_score(y_test, y_opt):.3f}")
        col3.metric("False Positives", f"{confusion_matrix(y_test, y_opt)[0,1]}")
        
        # Show which test samples are flagged at this threshold
        flagged_indices = np.where(y_opt == 1)[0]
        if len(flagged_indices) > 0:
            st.subheader(f"📋 Students Flagged at Threshold (Test Set) – {len(flagged_indices)} samples")
            flagged_df = pd.DataFrame({
                'Test Sample Index': flagged_indices,
                'True Label': y_test.iloc[flagged_indices].values,
                'Predicted Probability': y_proba[flagged_indices].round(3),
                'Prediction': y_opt[flagged_indices]
            })
            st.dataframe(flagged_df, width='stretch')
            st.caption("Note: These are test samples from the telemetry data, not the 200 synthetic students.")
        else:
            st.info("No test samples are flagged at this threshold.")

# ----------------------------------------------------------------------
# PAGE 7: VIOLATIONS
# ----------------------------------------------------------------------
elif page == "🚨 Violations":
    st.markdown("### 🚨 Violation Reports")
    cheaters = students_df[students_df['is_cheating'] == 1]
    if cheaters.empty:
        st.success("✅ No violations detected!")
    else:
        st.warning(f"🔴 {len(cheaters)} students flagged for suspicious behavior")
        st.dataframe(cheaters[['student_id', 'name', 'risk_category', 'risk_score', 'cheating_type', 'violation_count', 'violation_summary']],
                     width='stretch')
        for _, student in cheaters.iterrows():
            with st.expander(f"⚠️ Student {student['student_id']} - {student['name']}"):
                st.write(f"**Risk Score:** {student['risk_score']:.1f}")
                st.write(f"**Category:** {student['risk_category']}")
                st.write(f"**Cheating Type:** {student['cheating_type']}")
                st.write(f"**Violation Count:** {student['violation_count']}")
                st.write(f"**Violation Summary:** {student['violation_summary']}")
                if student['risk_category'] == 'High':
                    st.error("🔴 Immediate review required!")
                else:
                    st.warning("🟠 Review recommended within 24 hours.")

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Panopticon v1.0**  \n*EduGuard AI - ML Division*")