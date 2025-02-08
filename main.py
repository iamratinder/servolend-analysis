from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("api_key"))

def get_loan_info(user_data):
    model_input = {  # Prepare data for the model
        "age": user_data["age"],
        "income": user_data["income"],
        "ownership": user_data["ownership"],
        "employment_len": user_data["employment_len"],
        "loan_intent": user_data["loan_intent"],
        "loan_amnt": user_data["loan_amnt"],
        "loan_int_rate": user_data["loan_int_rate"],
        "loan_percent_income": user_data["loan_percent_income"],
        "cred_hist_len": user_data["cred_hist_len"]
    }

    response = requests.post(os.getenv("ml_key"), json=model_input)
    prob_eligible, prob_not_eligible = (response.json().get(k) for k in ["prob of eligible", "prob of not eligible"]) # Extract the prediction

    prompt = f"""
    A user has applied for a loan.  Their information is as follows:
    Age: {user_data["age"]}
    Income: {user_data["income"]}
    Ownership: {user_data["ownership"]}
    employment_len: {user_data["employment_len"]}
    loan_intent: {user_data["loan_intent"]}
    loan_amnt": {user_data["loan_amnt"]}
    loan_int_rate":{user_data["loan_int_rate"]} 
    loan_percent_income":{user_data["loan_percent_income"]}
    cred_hist_len":{user_data["cred_hist_len"]}


    The loan defaulter model predicted: {prob_eligible} and {prob_not_eligible} which is the prob of being eligible or not respectively

    Based on this prediction, provide information about their loan eligibility status, potential loan options, and next steps. Explain possible reasons and suggest improvements.
    Also note that do not include things like as prediction of model etc.... just provide your response as you only is analyzing and providing summary for the decision made by the model at the backend
    Also if the applicant is eligible means prediction is 0 (not defaulter) than in that case also tell congratulations you are likely to get the loan.
    """
    ans = llm.invoke(prompt)
    print(ans.content)


user_data = {
    "age": 21,
    "income": 960000,
    "ownership": "RENT",
    "employment_len": 5.0,
    "loan_intent": "PERSONAL",
    "loan_amnt": 10000000,
    "loan_int_rate": 11.14,
    "loan_percent_income": 1000,
    "cred_hist_len": 2
}
configure()
get_loan_info(user_data)
