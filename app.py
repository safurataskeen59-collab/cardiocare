import os
import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(
    page_title="CardioCare | Cardiovascular Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_cardio_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.pkl"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

EXPECTED_FEATURES = [
    "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "BMI"
]

# -------------------- STYLING --------------------
st.markdown(
    """
    <style>
    .main-title {font-size: 2.35rem; font-weight: 800; margin-bottom: 0.1rem;}
    .subtitle {font-size: 1rem; color: #64748b; margin-bottom: 1.2rem;}
    .notice {padding: 0.9rem 1rem; border-radius: 0.7rem;
             background: rgba(100,116,139,0.10); border-left: 4px solid #64748b;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.22);
        padding: 0.85rem; border-radius: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- FILE LOADING --------------------
@st.cache_resource
def load_model_and_artifacts():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        try:
            metadata = joblib.load(METADATA_PATH)
        except Exception:
            metadata = {}
    return model, features, metadata

def show_missing_artifacts():
    st.error("The trained model files are not available yet.")
    st.markdown(
        "Before using the prediction page, run `analysis.py` successfully so it creates:"
        "\n- `models/best_cardio_model.pkl`\n- `models/features.pkl`\n- `models/model_metadata.pkl` (optional)"
    )

st.sidebar.title("❤️ CardioCare")
st.sidebar.caption("Cardiovascular Disease Prediction")
st.sidebar.markdown("---")
st.sidebar.info(
    "Educational machine-learning project. Predictions are not medical advice "
    "and must not be used to diagnose or make treatment decisions."
)

st.markdown('<div class="main-title">CardioCare</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Cardiovascular disease prediction using machine learning</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="notice"><b>Important:</b> This prototype predicts the dataset label from '
    'the information entered. It is not clinically validated and cannot determine an '
    'individual’s actual medical condition.</div>',
    unsafe_allow_html=True,
)

prediction_tab, performance_tab, about_tab = st.tabs(
    ["🩺 Prediction", "📊 Model Performance", "ℹ️ About"]
)

# -------------------- PREDICTION TAB --------------------
with prediction_tab:
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        show_missing_artifacts()
    else:
        try:
            model, saved_features, metadata = load_model_and_artifacts()
            
            if list(saved_features) != EXPECTED_FEATURES:
                st.error("The saved model features do not match the app's expected features.")
                st.stop()

            best_model_name = metadata.get("best_model", "Trained model")

            st.caption(f"Model in use: {best_model_name}")

            with st.form("prediction_form"):
                st.subheader("Enter health-related information")
                st.caption("Enter the values requested below. All fields are required.")

                col1, col2, col3 = st.columns(3)

                with col1:
                    age = st.number_input(
                        "Age (years)", min_value=1, max_value=120, value=45
                    )
                    gender = st.selectbox("Gender", ["Female", "Male"])
                    height = st.number_input(
                        "Height (cm)", min_value=100.0, max_value=250.0,
                        value=165.0, step=1.0
                    )
                    weight = st.number_input(
                        "Weight (kg)", min_value=30.0, max_value=250.0,
                        value=65.0, step=1.0
                    )

                with col2:
                    ap_hi = st.number_input(
                        "Systolic blood pressure", min_value=60, max_value=250,
                        value=120
                    )
                    ap_lo = st.number_input(
                        "Diastolic blood pressure", min_value=40, max_value=200,
                        value=80
                    )
                    cholesterol = st.selectbox(
                        "Cholesterol level",
                        [1, 2, 3],
                        format_func=lambda x: {
                            1: "1 - Normal",
                            2: "2 - Above normal",
                            3: "3 - High"
                        }[x]
                    )
                    gluc = st.selectbox(
                        "Glucose level",
                        [1, 2, 3],
                        format_func=lambda x: {
                            1: "1 - Normal",
                            2: "2 - Above normal",
                            3: "3 - High"
                        }[x]
                    )

                with col3:
                    smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
                    alco = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
                    active = st.selectbox(
                        "Are you physically active?", ["Yes", "No"]
                    )

                submitted = st.form_submit_button(
                    "Predict", type="primary", use_container_width=True
                )

            if submitted:
                if ap_hi <= ap_lo:
                    st.error(
                        "Systolic blood pressure should be greater than "
                        "diastolic blood pressure. Please check your entries."
                    )
                else:
                    bmi = weight / ((height / 100) ** 2)

                    # Keep feature names and order consistent with analysis.py.
                    input_data = pd.DataFrame([{
                        "age_years": float(age),
                        "gender": 2 if gender == "Male" else 1,
                        "height": float(height),
                        "weight": float(weight),
                        "ap_hi": float(ap_hi),
                        "ap_lo": float(ap_lo),
                        "cholesterol": int(cholesterol),
                        "gluc": int(gluc),
                        "smoke": 1 if smoke == "Yes" else 0,
                        "alco": 1 if alco == "Yes" else 0,
                        "active": 1 if active == "Yes" else 0,
                        "BMI": float(bmi)
                    }], columns=EXPECTED_FEATURES)

                    try:
                        prediction = int(model.predict(input_data)[0])

                        st.subheader("Prediction result")
                        metric1, metric2 = st.columns(2)

                        metric1.metric("Calculated BMI", f"{bmi:.1f}")
                        metric2.metric("Model output", str(prediction))

                        if prediction == 1:
                            st.warning(
                                "The model predicts class 1 in the dataset, "
                                "which represents the cardiovascular disease "
                                "label. This is not a medical diagnosis."
                            )
                        else:
                            st.success(
                                "The model predicts class 0 in the dataset, "
                                "which represents the absence of the disease "
                                "label. This does not rule out a medical condition."
                            )

                        with st.expander("View the input sent to the model"):
                            st.dataframe(input_data, use_container_width=True)

                        st.info(
                            "This educational model is not clinically validated. "
                            "Do not use its prediction to make medical decisions. "
                            "Consult a qualified healthcare professional for "
                            "medical concerns."
                        )

                    except Exception as prediction_error:
                        st.error(f"Prediction could not be completed: {prediction_error}")

        except Exception as loading_error:
            st.error(f"Could not load the trained model: {loading_error}")
            st.info(
                "Check that analysis.py has completed successfully and that "
                "the required model files exist in the models folder."
            )

# -------------------- MODEL PERFORMANCE TAB --------------------
with performance_tab:
    st.header("Model Performance")
    st.write(
        "This section displays the evaluation results produced by analysis.py. "
        "The results will appear after the analysis has been run successfully."
    )

    model_results_path = RESULTS_DIR / "model_results.csv"
    cv_results_path = RESULTS_DIR / "cross_validation_results.csv"
    report_path = RESULTS_DIR / "classification_report.csv"

    if model_results_path.exists():
        results_df = pd.read_csv(model_results_path)

        st.subheader("Test-set model comparison")
        st.dataframe(results_df, use_container_width=True)

        if {"Model", "Accuracy"}.issubset(results_df.columns):
            st.subheader("Accuracy by model")
            accuracy_chart = results_df.set_index("Model")[["Accuracy"]]
            st.bar_chart(accuracy_chart)

    else:
        st.info(
            "Model comparison results are not available yet. "
            "They will be created when analysis.py runs successfully."
        )

    if cv_results_path.exists():
        st.subheader("5-fold cross-validation results")
        cv_df = pd.read_csv(cv_results_path)
        st.dataframe(cv_df, use_container_width=True)
    else:
        st.info("Cross-validation results have not been generated yet.")

    if report_path.exists():
        st.subheader("Classification report for the selected model")
        report_df = pd.read_csv(report_path, index_col=0)
        st.dataframe(report_df, use_container_width=True)

    st.subheader("Analysis plots")

    plot_files = [
        ("01_target_distribution.png", "Target distribution"),
        ("02_age_distribution.png", "Age distribution"),
        ("03_gender_distribution.png", "Gender distribution"),
        ("04_correlation_heatmap.png", "Feature correlation heatmap"),
        ("05_feature_distributions.png", "Feature distributions"),
        ("06_blood_pressure_distribution.png", "Blood pressure distribution"),
        ("07_bmi_distribution.png", "BMI distribution"),
        ("08_cholesterol_distribution.png", "Cholesterol distribution"),
        ("09_glucose_distribution.png", "Glucose distribution"),
        ("10_smoking_distribution.png", "Smoking distribution"),
        ("11_activity_distribution.png", "Physical activity distribution"),
        ("12_alcohol_distribution.png", "Alcohol consumption"),
        ("13_age_by_target.png", "Age by target"),
        ("14_bmi_by_target.png", "BMI by target"),
        ("15_confusion_matrix.png", "Confusion matrix"),
        ("16_roc_curves.png", "ROC curves"),
        ("17_model_comparison.png", "Model comparison"),
        ("18_feature_importance.png", "Feature importance"),
    ]

    available_plots = [
        (filename, title)
        for filename, title in plot_files
        if (PLOTS_DIR / filename).exists()
    ]

    if available_plots:
        for filename, title in available_plots:
            with st.expander(title, expanded=False):
                st.image(
                    str(PLOTS_DIR / filename),
                    caption=title,
                    use_container_width=True
                )
    else:
        st.info(
            "No analysis plots are available yet. They will appear here "
            "after analysis.py creates them."
        )

# -------------------- ABOUT TAB --------------------
with about_tab:
    st.header("About CardioCare")

    st.write(
        """
        CardioCare is an educational machine-learning project that analyses
        cardiovascular disease dataset records and compares classification models.
        """
    )

    st.subheader("Project workflow")
    st.markdown(
        """
        1. Load and inspect the dataset.
        2. Clean invalid records and handle duplicate rows.
        3. Calculate age in years and Body Mass Index (BMI).
        4. Create exploratory data analysis visualizations.
        5. Train and compare five machine-learning models.
        6. Evaluate models using test-set metrics and 5-fold cross-validation.
        7. Save the selected model, evaluation results, and plots.
        8. Use the Streamlit app to demonstrate predictions.
        """
    )

    st.subheader("Technologies used")
    st.markdown(
        """
        - **Python** — programming language
        - **Pandas and NumPy** — data handling and numerical calculations
        - **Matplotlib and Seaborn** — data visualization
        - **Scikit-learn** — machine-learning models and evaluation
        - **Joblib** — saving and loading the trained model
        - **Streamlit** — interactive web application
        """
    )

    st.subheader("Important medical disclaimer")
    st.warning(
        "CardioCare is an educational prototype, not a clinically validated "
        "medical tool. Its predictions may be wrong and must not be used to "
        "diagnose cardiovascular disease, determine treatment, or replace "
        "professional medical advice."
    )

st.sidebar.markdown("---")
st.sidebar.caption("CardioCare | Educational project")
