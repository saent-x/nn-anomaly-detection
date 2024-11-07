import os
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import datetime
from sklearn.metrics import classification_report
from io import StringIO

# Utility function to find and return available plot folders
def find_plot_folders():
    training_dirs = []
    evaluation_dirs = []

    # Use regular expressions to match the timestamped folder names
    training_pattern = re.compile(r"^training_plots_\d{8}-\d{6}$")
    evaluation_pattern = re.compile(r"^evaluation_plots_\d{8}-\d{6}$")

    for folder in os.listdir('.'):
        if training_pattern.match(folder):
            training_dirs.append(folder)
        elif evaluation_pattern.match(folder):
            evaluation_dirs.append(folder)

    # Sort by date, latest first
    training_dirs.sort(reverse=True)
    evaluation_dirs.sort(reverse=True)

    return training_dirs, evaluation_dirs


# Load metrics and plot paths from the selected directories
def load_metrics_and_plots(training_dir=None, evaluation_dir=None):
    metrics = {}
    evaluation_report = ""
    roc_curve_path = ""
    confusion_matrix_path = ""
    training_history_path = ""

    # Load training metrics and history if training_dir is provided
    if training_dir:
        metrics_file = os.path.join(training_dir, "metrics_summary.txt")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                for line in f:
                    key, value = line.strip().split(": ")
                    metrics[key] = float(value)
        
        training_history_path = os.path.join(training_dir, "training_history.png")

    # Load evaluation metrics, ROC curve, and confusion matrix if evaluation_dir is provided
    if evaluation_dir:
        evaluation_file = os.path.join(evaluation_dir, "classification_report.txt")
        if os.path.exists(evaluation_file):
            with open(evaluation_file, 'r') as f:
                evaluation_report = f.read()

        roc_curve_path = os.path.join(evaluation_dir, "roc_curve.png")
        confusion_matrix_path = os.path.join(evaluation_dir, "confusion_matrix.png")

    return metrics, evaluation_report, roc_curve_path, confusion_matrix_path, training_history_path


# Convert classification report text to a DataFrame
def parse_classification_report(report_text):
    report_data = []
    report = StringIO(report_text)
    for line in report:
        row = line.strip().split()
        if len(row) == 5 and row[0] != "accuracy":
            report_data.append(row)
    columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    return pd.DataFrame(report_data, columns=columns)


# Streamlit app layout
st.set_page_config(page_title="Model Training and Evaluation", layout="wide")

# Styling
st.markdown(
    """
    <style>
        .stMarkdown {font-size: larger;}
        .stDataFrame {font-size: larger;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Model Training and Evaluation Results")
st.write("This app displays the results of your model training and evaluation in an organized manner.")

# Find available plot folders
training_dirs, evaluation_dirs = find_plot_folders()

# Side tabs for Training Plots and Evaluation Plots
tab1, tab2, tab3 = st.tabs(["Training Plots", "Evaluation Plots", "Test Inference"])

with tab1:
    st.header("Training Plots")
    if training_dirs:
        selected_training_dir = st.selectbox("Select Training Plot Folder", training_dirs)
        metrics, _, _, _, training_history_path = load_metrics_and_plots(training_dir=selected_training_dir)

        # Display Training History Plot
        st.subheader("Training History")
        if os.path.exists(training_history_path):
            st.image(training_history_path, caption="Training and Validation Accuracy & Loss over Epochs")
        else:
            st.write("Training history plot not found.")

        # Display Final Training and Validation Metrics in a table format
        st.subheader("Summary of Training Metrics")
        if metrics:
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Final Training Accuracy",
                    "Final Validation Accuracy",
                    "Final Training Loss",
                    "Final Validation Loss",
                    "Best Validation Accuracy",
                    "Best Validation Loss",
                    "Epochs Trained"
                ],
                "Value": [
                    metrics.get('final_train_accuracy', 'N/A'),
                    metrics.get('final_val_accuracy', 'N/A'),
                    metrics.get('final_train_loss', 'N/A'),
                    metrics.get('final_val_loss', 'N/A'),
                    metrics.get('best_val_accuracy', 'N/A'),
                    metrics.get('best_val_loss', 'N/A'),
                    metrics.get('epochs_trained', 'N/A')
                ]
            })
            st.table(metrics_df)
        else:
            st.write("No training metrics available.")
        
        # Analysis and Interpretation Section
        st.header("Analysis and Interpretation")
        st.write("""
        This section provides a summary and interpretation of the model performance. 

        - The **training and validation accuracy/loss curves** indicate how well the model has generalized to new data.
        - The **ROC curve and AUC score** provide a measure of the model's ability to distinguish between positive and negative classes.
        - The **confusion matrix** shows the number of correct and incorrect predictions for each class, helping you identify specific areas for improvement.
        - The **classification report** gives precision, recall, and F1 scores for each class, offering insight into class-specific performance.
        """)

with tab2:
    st.header("Evaluation Plots")
    if evaluation_dirs:
        selected_evaluation_dir = st.selectbox("Select Evaluation Plot Folder", evaluation_dirs)
        _, evaluation_report, roc_curve_path, confusion_matrix_path, _ = load_metrics_and_plots(evaluation_dir=selected_evaluation_dir)

        # Display ROC Curve and Confusion Matrix side by side
        st.subheader("Model Evaluation")
        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists(roc_curve_path):
                st.image(roc_curve_path, caption="ROC Curve with AUC Score")
            else:
                st.write("ROC curve plot not found.")

        with col2:
            if os.path.exists(confusion_matrix_path):
                st.image(confusion_matrix_path, caption="Confusion Matrix")
            else:
                st.write("Confusion matrix plot not found.")

        # Display Classification Report as Table
        st.subheader("Classification Report")
        if evaluation_report:
            report_df = parse_classification_report(evaluation_report)
            st.table(report_df)
        else:
            st.write("No classification report available.")
            
        # Analysis and Interpretation Section
        st.header("Analysis and Interpretation")
        st.write("""
        This section provides a summary and interpretation of the model performance. 

        - The **training and validation accuracy/loss curves** indicate how well the model has generalized to new data.
        - The **ROC curve and AUC score** provide a measure of the model's ability to distinguish between positive and negative classes.
        - The **confusion matrix** shows the number of correct and incorrect predictions for each class, helping you identify specific areas for improvement.
        - The **classification report** gives precision, recall, and F1 scores for each class, offering insight into class-specific performance.
        """)
        
with tab3:
    st.header("Test Inference")

