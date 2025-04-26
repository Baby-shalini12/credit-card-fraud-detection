import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection")

st.info("🔒 Your uploaded data is used only for analysis and is not displayed or stored.")

uploaded_file = st.file_uploader("Upload your credit_card.csv file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Only remove rows where the target is NaN
    df = df.dropna(subset=['Class'])

    # Features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Check if both classes have at least 2 samples
    if y.value_counts().min() < 2:
        st.error("Not enough samples in one of the classes to perform stratified split and SMOTE. Please check your dataset.")
        st.stop()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to balance the classes in the training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_res, y_train_res)

    # Predict on test set
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, digits=4))

    roc_auc = roc_auc_score(y_test, y_proba)
    st.write(f"**ROC-AUC Score:** {roc_auc:.4f}")

    st.subheader("ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax2.plot([0,1], [0,1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("Please upload your credit_card.csv file to begin.")

