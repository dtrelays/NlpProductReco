import streamlit as st
import pandas as pd
import os
from src.utils import load_object


from src.pipeline.predict_pipeline import CustomData,PredictPipeline

@st.cache_resource
def load_model():

    model_path=os.path.join("artifacts","model.pkl")
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

    model=load_object(file_path=model_path)

    preprocessor=load_object(file_path=preprocessor_path)

    loaded_model = model["model"]
    loaded_params = model["params"]

    # Create a new instance of the model with the loaded hyperparameters
    model_with_loaded_params = loaded_model.set_params(**loaded_params)            

    return model_with_loaded_params,preprocessor


model,preprocessor=load_model()

def main():

    intent_type = ['PERSONAL','EDUCATION','MEDICAL','VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION']
    
    home_type = ['RENT','OWN','MORTGAGE','OTHER']
    
    # Set a background color
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F5F5F5;
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

#   <div style="background-color: #5CDB95; padding: 20px; text-align: center;">
#         <h1 style="color: #EDF5E1; font-size: 26px; font-weight: bold; text-transform: uppercase;">Credit Risk Prediction App</h1>
#     </div>

    html_temp = """
    <div style="background-color: #FF725C; padding: 20px; text-align: center;">
        <h1 style="color: #FFFFFF; font-size: 26px; font-weight: bold; text-transform: uppercase;">Credit Risk Prediction App</h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    # Custom styling for the subheader
    html_temp_header = """
        <div style="padding: 16px;text-align:left;">
            <h3 style="font-weight: bold; font-size: 20px;">Fill the below Form to Know Customer Default Probability:</h3>
        </div>
        """
    st.markdown(html_temp_header, unsafe_allow_html=True)

    # Demographic info
    st.subheader("Demographic Info")

    # Custom styling for the input fields
    input_style = """
        <style>
        .stTextInput, .stNumberInput, .stSelectbox {
            background-color: #FFFFFF;
            color: #333333;
            font-size: 16px;
            font-weight: bold;
            padding: 8px;
            border-radius: 8px;
            box-shadow: none;
            border: 1px solid #111111;
            margin-bottom: 16px;
        }
        .stTextInput:focus, .stNumberInput:focus, .stSelectbox:focus {
            border: 2px solid #05386B;
            box-shadow: none;
        }
        </style>
        """
    st.markdown(input_style, unsafe_allow_html=True)

    name = st.text_input("Customer Name",value="Mohit Gupta")
    contact_number = st.text_input("Mobile Number",value="9876543210")
    age = st.number_input("Age", 18, 75, 25)
    selected_home = st.selectbox("Home Type", sorted(home_type), index=0)
    income = st.slider("Income (USD)", 1000, 2000000, 5000)
    selected_intent = st.selectbox("Loan Purpose", sorted(intent_type), index=0)
    loan_amount = st.number_input("Loan Amount Required", 500, 300000)
    interest_rate = st.number_input("Interest Rate", 6.0, 30.0, 10.0)
    credit_history_length = st.slider("Credit History (Years)", 0, 30, 5)
    employment_length = st.slider("Employment History (Years)", 0, 40, 5)

    # Validation logic
    if len(contact_number) > 10:
        st.error("Error: Mobile Number cannot be greater than 10 digits")
        st.stop()

    # Submit button
    if st.button("Check Default Chance"):

        data = CustomData(
            age=age,
            income=income,
            credit_history_length=credit_history_length,
            loan_amount=loan_amount,
            interest_rate=interest_rate,
            employment_length=employment_length,
            home_type=selected_home,
            intent_type=selected_intent
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df, model, preprocessor)

        # Custom styling for the text
        text_style = """
            <style>
            .stText {
                font-size: 16px;
                font-weight: bold;
                color: #333333;
                margin-top: 16px;
                margin-bottom: 16px;
            }
            </style>
            """
        st.markdown(text_style, unsafe_allow_html=True)

        # Display the score
        st.write(f"<p class='stText'>Getting Loan Default Chance for: {name}</p>", unsafe_allow_html=True)

        if results == 'Low':
            st.success("Chances of default are low, you can approve the loan")
        else:
            st.error("Chances of default are high, you should reject the loan")

    # Reset button
    if st.button("Reset"):
        # Clear all inputs and selections
        st.experimental_rerun()


if __name__=="__main__":
    main()
