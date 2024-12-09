# # Gradio App for Predictive Model Deployment
# This app predicts whether a client will subscribe to a term deposit.

import pandas as pd
import joblib
import gradio as gr

# Load the trained model
model = joblib.load("model/bank_term_deposit_model.pkl")

# Define categorical and numerical features
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# Define the prediction function
def predict_term_deposit(age, job, marital, education, default, balance, housing, loan, contact, month, duration, campaign, pdays, previous, poutcome):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])
    
    # Predict probability and class
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of subscribing
    
    # Map prediction to human-readable output
    result = "Yes" if prediction[0] == 1 else "No"
    return f"Subscription: {result}", f"Probability of Subscription: {prediction_proba:.2f}"

# Gradio interface
inputs = [
    gr.Number(label="Age"),
    gr.Dropdown(['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur',
                        'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'], label="Job"),
    gr.Dropdown(['married', 'divorced', 'single'], label="Marital Status"),
    gr.Dropdown(['unknown', 'secondary', 'primary', 'tertiary'], label="Education"),
    gr.Radio(['yes', 'no'], label="Has Credit in Default?"),
    gr.Number(label="Balance"),
    gr.Radio(['yes', 'no'], label="Has Housing Loan?"),
    gr.Radio(['yes', 'no'], label="Has Personal Loan?"),
    gr.Dropdown(['unknown', 'telephone', 'cellular'], label="Contact Communication Type"),
    gr.Dropdown(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], label="Last Contact Month"),
    gr.Number(label="Duration (seconds)"),
    gr.Number(label="Number of Contacts During Campaign"),
    gr.Number(label="Number of Days Since Last Contact (-1 means never contacted)"),
    gr.Number(label="Number of Contacts Before Campaign"),
    gr.Dropdown(['unknown', 'other', 'failure', 'success'], label="Outcome of Previous Campaign")
]

outputs = [
    gr.Textbox(label="Prediction"),
    gr.Textbox(label="Probability")
]

app = gr.Interface(
    fn=predict_term_deposit,
    inputs=inputs,
    outputs=outputs,
    title="Bank Term Deposit Prediction",
    description="Provide client and campaign details to predict whether the client will subscribe to a term deposit.",
)

# Launch the app
app.launch()
