import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import shap
import re
from password_strength import PasswordPolicy
import hashlib
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

# ---------------------- Database Setup ----------------------
def init_db():
    conn = sqlite3.connect('ckd_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  email TEXT UNIQUE,
                  password_hash TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()
init_db()

# ---------------------- Password Hashing ----------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------------- App Configuration ----------------------
st.set_page_config(
    page_title="CKD Predictor Pro",
    layout="wide",
    page_icon="🩺",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# CKD Predictor Pro\nAdvanced Chronic Kidney Disease prediction tool"
    }
)

# ---------------------- Password Policy ----------------------
policy = PasswordPolicy.from_names(
    length=8,
    uppercase=1,
    numbers=1,
    special=1,
)

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def check_password_strength(password):
    return policy.test(password)

# ---------------------- User Authentication Functions ----------------------
def register_user(username, email, password):
    conn = sqlite3.connect('ckd_users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                  (username, email, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            st.error("Username already exists!")
        elif "email" in str(e):
            st.error("Email already registered!")
        return False
    finally:
        conn.close()

def verify_user(email, password):
    conn = sqlite3.connect('ckd_users.db')
    c = conn.cursor()
    c.execute("SELECT username, password_hash FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    if result and result[1] == hash_password(password):
        return result[0]
    return None

def show_auth_forms():
    if not st.session_state.get("authenticated", False):
        st.title("CKD Predictor Pro Login")
        tabs = st.tabs(["Login", "Register"])
        with tabs[0]:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    username = verify_user(email, password)
                    if username:
                        st.session_state.authenticated = True
                        st.session_state.current_user = {'username': username, 'email': email}
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        with tabs[1]:
            with st.form("register_form"):
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Register")
                if submitted:
                    errors = []
                    if not username:
                        errors.append("Username is required")
                    if not email:
                        errors.append("Email is required")
                    elif not validate_email(email):
                        errors.append("Invalid email format")
                    if not password:
                        errors.append("Password is required")
                    elif password != confirm_password:
                        errors.append("Passwords don't match")
                    elif check_password_strength(password):
                        errors.append("Password too weak - needs uppercase, number, and special char")
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        if register_user(username, email, password):
                            st.success("Registration successful! Please login.")
                            st.rerun()
        st.stop()
    else:
        st.sidebar.success(f"Logged in as {st.session_state.current_user['username']}")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()

# ---------------------- Data Loading ----------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("kidney_disease.csv")
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/patilgirish815/Kidney_Cancer_Prediction_Using_Machine_Learning/main/dataset/kidney_disease.csv"
        df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["id"], errors='ignore')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col].astype(str))
    df['classification'] = df['classification'].replace({0: 0, 2: 1, 'ckd': 1, 'notckd': 0, 'yes': 1, 'no': 0})
    return df

# ---------------------- Model Building ----------------------
def build_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', use_label_encoder=False)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# ---------------------- Plotting Functions ----------------------
def plot_confusion_matrix(cm, title):
    if cm.shape == (2, 2):
        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['No CKD', 'CKD'], y=['No CKD', 'CKD'],
                        color_continuous_scale='viridis', text_auto=True)
        fig.update_layout(title_text=title)
        return fig
    else:
        st.error(f"Unexpected confusion matrix shape: {cm.shape}. Cannot plot.")
        return None

def plot_accuracy_bar(metrics):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(metrics.keys()),
        y=[metrics[m]["Accuracy"]*100 for m in metrics],
        text=[f"{metrics[m]['Accuracy']*100:.2f}%" for m in metrics],
        textposition='auto',
        marker_color=['#1976d2', '#43a047']
    ))
    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100])
    )
    return fig

# ---------------------- Main App Flow ----------------------
show_auth_forms()
st.title("Chronic Kidney Disease Predictor Pro")
st.write(f"Welcome back, {st.session_state.current_user['username']}!")

# Data loading
with st.spinner("Loading data..."):
    df = load_data()
if df is None:
    st.stop()

X = df.drop("classification", axis=1)
y = df["classification"]

# Data scaling
with st.spinner("Scaling data..."):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Train-test split (fixed test size, e.g., 20%)
with st.spinner("Splitting data..."):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    except ValueError as e:
        st.error(f"Invalid test size or insufficient data for stratification: {str(e)}")
        st.stop()

# Always retrain models to avoid feature mismatch
with st.spinner("Training models..."):
    models = build_models(X_train, y_train)
selected_features = X.columns  # All features

# ---------------------- Patient Data Entry ----------------------
st.subheader("Enter Patient Data")
with st.form("predict_form"):
    patient_name = st.text_input("Patient Name (required)")
    patient_age = st.number_input("Patient Age (required)", min_value=0, max_value=120, value=0, step=1)
    cols = st.columns(3)
    input_data = {}
    for i, feature in enumerate(selected_features):
        if feature.lower() in ["age", "name", "patient_name", "patient_age"]:
            continue  # skip if already handled
        col = cols[i % 3]
        with col:
            if df[feature].dtype == 'object' or df[feature].nunique() < 5:
                options = df[feature].unique()
                default_index = 0
                try:
                    default_value = df[feature].mode()[0]
                    if default_value in options:
                        default_index = list(options).index(default_value)
                except:
                    pass
                input_data[feature] = st.selectbox(
                    feature,
                    options,
                    index=default_index
                )
            else:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                step = 0.01 if df[feature].dtype == 'float' else 1.0
                input_data[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=min_val,
                    step=step,
                    format="%.2f" if step == 0.01 else "%.0f"
                )
    submitted = st.form_submit_button("Predict CKD Risk", type="primary")

if submitted:
    if not patient_name or patient_age is None:
        st.error("Please enter both Patient Name and Age.")
    else:
        st.subheader("Prediction Results")
        # Insert patient age into input_data if 'age' is a feature
        if 'age' in selected_features:
            input_data['age'] = patient_age
        # Order input values to match model features
        input_values = [input_data[feature] if feature != 'age' else patient_age for feature in selected_features]
        user_input = np.array(input_values).reshape(1, -1)
        user_scaled = scaler.transform(user_input)
        results = {}
        for model_name, model in models.items():
            prediction = model.predict(user_scaled)[0]
            probability = model.predict_proba(user_scaled)[0][1]
            results[model_name] = (prediction, probability)
        # Show results for both models
        col_rf, col_xgb = st.columns(2)
        for col, model_name in zip([col_rf, col_xgb], ["Random Forest", "XGBoost"]):
            with col:
                pred, prob = results[model_name]
                st.markdown(f"**{model_name}**")
                if pred == 1:
                    st.markdown(f"Result: CKD Detected")
                else:
                    st.markdown(f"Result: No CKD Detected")
                st.markdown(f"Confidence: {prob*100:.2f}%")
                st.markdown(f"Patient: {patient_name}, Age: {patient_age}")
                st.markdown("Feature Importance (SHAP):")
                explainer = shap.TreeExplainer(models[model_name])
                shap_values = explainer.shap_values(user_scaled)
                plt.figure(figsize=(8, 2))
                shap.summary_plot(shap_values, user_scaled, feature_names=selected_features, show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.close()
        # Final summary for patient
        best_model = max(results.items(), key=lambda x: x[1][1])[0]
        best_pred, best_prob = results[best_model]
        st.info(f"Summary for {patient_name} (Age {patient_age}): "
                f"{'CKD Detected' if best_pred == 1 else 'No CKD Detected'} "
                f"with {best_model} (Confidence: {best_prob*100:.2f}%)")

# ---------------------- Model Comparison ----------------------
st.subheader("Model Comparison on Test Data")
metrics = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc_score = 0.0
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
    metrics[model_name] = {
        "Accuracy": accuracy,
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "F1-Score": report['1']['f1-score'],
        "AUC": auc_score
    }

df_metrics = pd.DataFrame(metrics).T
st.dataframe(df_metrics.style.format("{:.2%}", subset=["Accuracy", "Precision", "Recall", "F1-Score"]).format("{:.2f}", subset=["AUC"]))

# Show model comparison as a bar chart (poll style)
fig_acc = plot_accuracy_bar(metrics)
st.plotly_chart(fig_acc, use_container_width=True)

# Show which model is best
best_model_name = df_metrics["Accuracy"].idxmax()
best_acc = df_metrics.loc[best_model_name, "Accuracy"]
st.success(f"Best Model: {best_model_name} (Accuracy: {best_acc:.2%})")

# Confusion Matrix
st.write("Confusion Matrix")
cm_rf = confusion_matrix(y_test, models["Random Forest"].predict(X_test))
cm_xgb = confusion_matrix(y_test, models["XGBoost"].predict(X_test))
col1, col2 = st.columns(2)
with col1:
    st.write("Random Forest Confusion Matrix")
    fig_cm_rf = plot_confusion_matrix(cm_rf, "Random Forest")
    if fig_cm_rf:
        st.plotly_chart(fig_cm_rf)
with col2:
    st.write("XGBoost Confusion Matrix")
    fig_cm_xgb = plot_confusion_matrix(cm_xgb, "XGBoost")
    if fig_cm_xgb:
        st.plotly_chart(fig_cm_xgb)

# ---------------------- Data Exploration ----------------------
st.subheader("Data Exploration")
if st.checkbox("Show raw data"):
    st.dataframe(df)
if st.checkbox("Show feature distributions"):
    feature = st.selectbox("Select feature", df.columns)
    fig, ax = plt.subplots(figsize=(8, 4))
    if df[feature].nunique() < 10:
        df[feature].value_counts().plot(kind='bar', ax=ax, color='#4fc3f7')
    else:
        df[feature].plot(kind='hist', bins=20, ax=ax, color='#4fc3f7')
    ax.set_title(f"Distribution of {feature}", pad=15)
    ax.grid(False)
    st.pyplot(fig)
    plt.close(fig)