# 🩺 CKD Predictor Pro

![alt text](image.png)Chronic Kidney Disease prediction system with explainable AI and secure user authentication.

## 🌟 Features

### 🔐 Authentication
- User registration with email validation
- Secure password hashing (SHA-256)
- Login/logout functionality
- Session management

### 🧠 Prediction Engine
- Dual-model approach (Random Forest + XGBoost)
- SHAP value explanations
- Confidence visualization
- Feature importance analysis

### 📊 Dashboard
- Interactive input forms
- Real-time results
- Model performance metrics
- Confusion matrices
- Mobile-responsive design

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ckd-predictor-pro.git
cd ckd-predictor-pro

pip install -r requirements.txt
```

## 🖥️ Usage
Run the application:
```bash
streamlit run ckd_app_frontend.py
```