from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Load environment variables
def configure():
    load_dotenv()

# Define the data model
class UserData(BaseModel):
    age: float
    income: float
    employment_len: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cred_hist_len: float
    ownership: str
    loan_intent: str

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("API_KEY"))

@app.post('/analyse')
def analyse(data: UserData):
    # Prepare input for the ML model
    model_input = {
        "age": data.age,
        "income": data.income,
        "employment_len": data.employment_len,
        "loan_amnt": data.loan_amnt,
        "loan_int_rate": data.loan_int_rate,
        "loan_percent_income": data.loan_percent_income,
        "cred_hist_len": data.cred_hist_len,
        "ownership": data.ownership,
        "loan_intent": data.loan_intent
    }

    # Call the ML model API
    ml_endpoint = os.getenv("ml_key")  # Ensure this environment variable is set correctly
    response = requests.post(ml_endpoint, json=model_input)

    if response.status_code != 200:
        return {"error": "Failed to fetch loan prediction. Please try again."}

    # Extract prediction probabilities from API response
    response_data = response.json()
    prob_eligible = response_data.get("prob_eligible", 0)  # Default to 0 if missing
    prob_not_eligible = response_data.get("prob_not_eligible", 0)

    # Generate response using LLM
    prompt = f"""
    A user has applied for a loan.  Their information is as follows:
    Age: {data.age}
    Income: {data.income}
    Ownership: {data.ownership}
    Employment Length: {data.employment_len}
    Loan Intent: {data.loan_intent}
    Loan Amount: {data.loan_amnt}
    Loan Interest Rate: {data.loan_int_rate}
    Loan Percent Income: {data.loan_percent_income}
    Credit History Length: {data.cred_hist_len}

    The loan defaulter model predicted: {prob_eligible} and {prob_not_eligible} which is the prob of being eligible or not respectively

    Based on this prediction, provide information about their loan eligibility status, potential loan options, and next steps. Explain possible reasons and suggest improvements.
    Also note that do not include things like as prediction of model etc.... just provide your response as you only is analyzing and providing summary for the decision made by the model at the backend
    Also if the applicant is eligible means prediction is 0 (not defaulter) than in that case also tell congratulations you are likely to get the loan.
    """

    ans = llm.invoke(prompt)
    return {"message": ans.content}

# Run FastAPI server
configure()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
