import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import shap
import re
from password_strength import PasswordPolicy
import hashlib
import sqlite3
import base64

# Database setup
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

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# App configuration
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

# Dark theme CSS
dark_theme_css = """
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: white;
    }
    
    .dark-auth-container {
        background-color: #1e1e1e !important;
        color: white !important;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin: 2rem auto;
        max-width: 500px;
        border: 1px solid #333;
    }
    
    .dark-auth-container h3 {
        color: #4fc3f7 !important;
    }
    
    .dark-auth-container .stTextInput>div>div>input {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
    
    .dark-auth-container .stTextInput>div>div>input::placeholder {
        color: #aaa !important;
    }
    
    .dark-auth-container .stButton>button {
        background-color: #4fc3f7 !important;
        color: #121212 !important;
        font-weight: 600;
    }
    
    .dark-auth-container .stButton>button:hover {
        background-color: #3fb3e6 !important;
    }
    
    .dark-auth-container .stMarkdown {
        color: white !important;
    }
    
    .password-strength {
        margin-top: 8px;
        height: 6px;
        border-radius: 3px;
        background-color: #333 !important;
        overflow: hidden;
    }
    
    .password-strength-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s, background-color 0.3s;
    }
    
    .main-content {
        background-color: #1e1e1e;
        color: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background-color: #2d2d2d !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .no-ckd {
        color: #81c784 !important;
        font-weight: 600;
    }
    
    .ckd-detected {
        color: #ff8a65 !important;
        font-weight: 600;
    }
    
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #2a7f62, #ffc107, #dc3545);
        margin: 1rem 0;
        position: relative;
    }
    
    .confidence-marker {
        position: absolute;
        top: -3px;
        width: 3px;
        height: 16px;
        background-color: white;
    }
    
    .stSidebar {
        background-color: #1e1e1e !important;
        color: white !important;
    }
    
    .stSlider>div>div>div>div {
        background-color: #4fc3f7 !important;
    }
    
    .stCheckbox>div>label {
        color: white !important;
    }
    
    .stAlert {
        background-color: #333 !important;
        color: white !important;
    }
</style>
"""

# Apply dark theme
st.markdown(dark_theme_css, unsafe_allow_html=True)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'register_mode' not in st.session_state:
    st.session_state.register_mode = False

# Password policy
policy = PasswordPolicy.from_names(
    length=8,
    uppercase=1,
    numbers=1,
    special=1,
)

# Validation functions
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def check_password_strength(password):
    return policy.test(password)

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
        return result[0]  # Return username
    return None

# Authentication UI with dark theme
def show_auth_forms():
    if not st.session_state.authenticated:
        # Center the auth form
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            with st.container():
                st.markdown('<div class="dark-auth-container">', unsafe_allow_html=True)
                
                if st.session_state.register_mode:
                    st.markdown("### 🩺 Create Account")
                    st.markdown("---")
                    
                    with st.form("register_form"):
                        username = st.text_input("Username", key="reg_username")
                        email = st.text_input("Email", key="reg_email")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            password = st.text_input("Password", type="password", key="reg_password")
                        with col2:
                            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
                        
                        # Password strength meter
                        if password:
                            strength = 1 - (len(check_password_strength(password)) / 4)
                            strength_color = "#f44336" if strength < 0.5 else "#ffeb3b" if strength < 0.8 else "#4caf50"
                            st.markdown(f"""
                                <div class="password-strength">
                                    <div class="password-strength-fill" style="width: {strength*100}%; background-color: {strength_color};"></div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        submitted = st.form_submit_button("Create Account", type="primary")
                        
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
                                    st.session_state.authenticated = True
                                    st.session_state.current_user = {
                                        'username': username,
                                        'email': email
                                    }
                                    st.session_state.register_mode = False
                                    st.rerun()
                    
                    st.markdown("Already have an account?")
                    if st.button("Login instead"):
                        st.session_state.register_mode = False
                        st.rerun()
                
                else:  # Login mode
                    st.markdown("### 🔑 Login to CKD Predictor")
                    st.markdown("---")
                    
                    with st.form("login_form"):
                        email = st.text_input("Email", key="login_email")
                        password = st.text_input("Password", type="password", key="login_password")
                        
                        submitted = st.form_submit_button("Login", type="primary")
                        
                        if submitted:
                            username = verify_user(email, password)
                            if username:
                                st.session_state.authenticated = True
                                st.session_state.current_user = {
                                    'username': username,
                                    'email': email
                                }
                                st.rerun()
                            else:
                                st.error("Invalid credentials")
                    
                    st.markdown("Don't have an account?")
                    if st.button("Create account"):
                        st.session_state.register_mode = True
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.sidebar:
            st.success(f"Logged in as {st.session_state.current_user['username']}")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.rerun()

# Main app functionality
@st.cache_data
def load_data():
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
        
    return df

def select_features(X, y, k=15):
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features, selector

def build_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        with open(os.path.join(MODEL_SAVE_PATH, f"{name.lower().replace(' ', '_')}.pkl"), "wb") as f:
            pickle.dump(model, f)
    
    return models

def explain_prediction(model, input_data, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], input_data, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
    
    st.pyplot(plt.gcf(), bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["No CKD", "CKD"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["No CKD", "CKD"])
    return fig

# App layout
MODEL_SAVE_PATH = "saved_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Show authentication forms
show_auth_forms()

# Main app content
if not st.session_state.authenticated:
    st.stop()

# Main content container
st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.title("🩺 Chronic Kidney Disease Predictor Pro")
st.write(f"Welcome back, {st.session_state.current_user['username']}!")

with st.sidebar:
    st.header("⚙️ Model Settings")
    test_size = st.slider("Test set size (%)", 10, 40, 25)
    feature_selection = st.checkbox("Enable feature selection", True)
    num_features = st.slider("Number of features to select", 5, 25, 15, disabled=not feature_selection)

# Load data and prepare models
df = load_data()
X = df.drop("classification", axis=1)
y = df["classification"]

if feature_selection:
    X_selected, selected_features, selector = select_features(X, y, num_features)
    X = X[selected_features]
else:
    X_selected = X.values
    selected_features = X.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=test_size/100, 
    random_state=42,
    stratify=y
)

try:
    models = {}
    for name in ["Random Forest", "XGBoost"]:
        model_file = os.path.join(MODEL_SAVE_PATH, f"{name.lower().replace(' ', '_')}.pkl")
        if os.path.exists(model_file):
            with open(model_file, "rb") as f:
                models[name] = pickle.load(f)
    
    if len(models) != 2:
        models = build_models(X_train, y_train)
except:
    models = build_models(X_train, y_train)

# Prediction form
with st.form("predict_form"):
    st.subheader("🧪 Enter Patient Data")
    cols = st.columns(3)
    input_data = {}
    
    for i, feature in enumerate(selected_features):
        col = cols[i % 3]
        input_data[feature] = col.number_input(label=feature, value=float(X[feature].median()), step=0.01)
    
    submitted = st.form_submit_button("Predict CKD Risk")

if submitted:
    input_values = [input_data[feature] for feature in selected_features]
    user_input = np.array(input_values).reshape(1, -1)
    user_scaled = scaler.transform(user_input)
    
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        pred = model.predict(user_scaled)[0]
        proba = model.predict_proba(user_scaled)[0][1]
        predictions[name] = pred
        probabilities[name] = proba
    
    st.subheader("📊 Prediction Results")
    cols = st.columns(len(models))
    for (name, pred), col in zip(predictions.items(), cols):
        proba = probabilities[name]
        certainty_pos = (1 - proba) * 100 if pred == 0 else proba * 100
        
        with col:
            st.markdown(f"""
            <div class="prediction-card">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 2rem; margin-right: 1rem;">{'✅' if pred == 0 else '⚠️'}</div>
                    <h3>{name}</h3>
                </div>
                <div class="{'no-ckd' if pred == 0 else 'ckd-detected'}">
                    {'No CKD detected' if pred == 0 else 'CKD detected'}
                </div>
                <div>Confidence: {proba*100:.1f}%</div>
                <div class="confidence-bar">
                    <div class="confidence-marker" style="left: {certainty_pos}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    best_model_name = max(models.keys(), key=lambda x: models[x].score(X_test, y_test))
    st.subheader("🔍 Explanation")
    explain_prediction(models[best_model_name], user_scaled, selected_features)

# Model performance section
st.subheader("📈 Model Performance")
fig, ax = plt.subplots(figsize=(8, 5))
accuracies = [models[name].score(X_test, y_test)*100 for name in models]
ax.bar(models.keys(), accuracies, color=['#4fc3f7', '#81c784'])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
st.pyplot(fig)

st.write("Confusion Matrices:")
cols = st.columns(len(models))
for name, col in zip(models.keys(), cols):
    cm = confusion_matrix(y_test, models[name].predict(X_test))
    fig = plot_confusion_matrix(cm, name)
    col.pyplot(fig)

# Close main content container
st.markdown('</div>', unsafe_allow_html=True)