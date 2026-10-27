
import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

st.title(" Heart Disease Prediction System")
st.caption("Machine Learning Mini-Project | MCA")

ARTIFACTS = "artifacts"

# Check that the training script has been run
metadata_path = os.path.join(ARTIFACTS, "metadata.json")
comparison_path = os.path.join(
    ARTIFACTS, "model_comparison.csv"
)

if not os.path.exists(metadata_path):
    st.error(
        "Models are not trained yet. Run "
        "'python train_model.py' first."
    )
    st.stop()

with open(metadata_path, "r") as file:
    metadata = json.load(file)

features = metadata["features"]

model_files = {
    "Logistic Regression":
        "logistic_regression.joblib",
    "KNN": "knn.joblib",
    "SVM": "svm.joblib",
    "Naive Bayes": "naive_bayes.joblib"
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict",
        "Model Comparison",
        "Graphs"
    ]
)

if page == "Home":
    st.header("Project Overview")

    st.write(
        "This project demonstrates how supervised machine "
        "learning algorithms can classify a heart disease "
        "dataset into two target classes."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Records", metadata["dataset_rows"])
    col2.metric("Algorithms", len(model_files))
    col3.metric("Test Split", "20%")

    st.subheader("Algorithms Used")
    st.markdown("""
    - **Logistic Regression:** baseline classification
    - **KNN:** classification based on nearby training examples
    - **SVM:** classification using a separating boundary
    - **Naive Bayes:** probabilistic classification
    """)

    st.subheader("Project Workflow")
    st.write(
        "Dataset → Preprocessing → Train/Test Split → "
        "Model Training → Evaluation → Prediction"
    )

elif page == "Predict":
    st.header("Enter Dataset Features")
    st.write(
        "Enter values in the same format and units as the "
        "training dataset. The model uses the selected "
        "features to produce an educational classification."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)", 1, 120, 50
            )
            sex = st.selectbox(
                "Sex (0/1)", [0, 1],
                help="Use the dataset's coding."
            )
            cp = st.selectbox(
                "Chest pain type (cp)", [0, 1, 2, 3]
            )
            trestbps = st.number_input(
                "Resting blood pressure", 50, 250, 120
            )
            chol = st.number_input(
                "Cholesterol", 50, 700, 200
            )

        with col2:
            fbs = st.selectbox(
                "Fasting blood sugar > 120 (fbs)", [0, 1]
            )
            restecg = st.selectbox(
                "Resting ECG (restecg)", [0, 1, 2]
            )
            thalach = st.number_input(
                "Maximum heart rate (thalach)",
                40, 250, 150
            )
            exang = st.selectbox(
                "Exercise-induced angina (exang)", [0, 1]
            )
            oldpeak = st.number_input(
                "Oldpeak (ST depression)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        with col3:
            slope = st.selectbox(
                "ST slope", [0, 1, 2]
            )
            ca = st.selectbox(
                "Major vessels (ca)", [0, 1, 2, 3, 4]
            )
            thal = st.selectbox(
                "Thal value", [0, 1, 2, 3]
            )
            selected_model = st.selectbox(
                "Select ML Algorithm",
                list(model_files.keys())
            )

        submitted = st.form_submit_button(
            "Run Prediction",
            use_container_width=True
        )

    if submitted:
        model_path = os.path.join(
            ARTIFACTS, model_files[selected_model]
        )

        if not os.path.exists(model_path):
            st.error("Model file missing. Run train_model.py.")
            st.stop()

        model = load_model(model_path)

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }], columns=features)

        prediction = int(model.predict(input_data)[0])

        st.subheader("Model Output")

        if prediction == 1:
            st.error(
                "Predicted class: 1 — positive class "
                "in the training dataset."
            )
        else:
            st.success(
                "Predicted class: 0 — negative class "
                "in the training dataset."
            )

        st.caption(
            "This output is a model classification, not a "
            "diagnosis, risk assessment, or assurance that "
            "heart disease is present or absent."
        )

        st.write("Selected algorithm:", selected_model)

elif page == "Model Comparison":
    st.header("Model Performance Comparison")

    if os.path.exists(comparison_path):
        results = pd.read_csv(comparison_path)
        st.dataframe(
            results.round(3),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            results.set_index("Model")[
                ["Accuracy", "Precision", "Recall", "F1 Score"]
            ]
        )

        st.caption(
            "Scores are from a single held-out test split. "
            "They do not establish clinical reliability."
        )
    else:
        st.error("Run train_model.py to generate results.")

elif page == "Graphs":
    st.header("Data Analysis and Evaluation Graphs")

    graphs = {
        "Model Comparison":
            "model_comparison.png",
        "Correlation Heatmap":
            "correlation_heatmap.png",
        "ROC Curves":
            "roc_comparison.png"
    }

    for title, filename in graphs.items():
        path = os.path.join(
            ARTIFACTS, "plots", filename
        )
        st.subheader(title)
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Graph not found. Run train_model.py.")

    st.subheader("Confusion Matrices")
    for name in model_files:
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(
            ARTIFACTS, "plots",
            f"{safe_name}_confusion_matrix.png"
        )
        if os.path.exists(path):
            st.write(name)
            st.image(path, width=500)
