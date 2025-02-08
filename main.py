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
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("api_key"))

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
You are a highly experienced financial advisor reviewing a loan application. The following details are provided for your assessment:

### Applicant Information:
- **Age:** {data.age}
- **Income:** ${data.income}
- **Ownership Status:** {data.ownership}
- **Employment Length:** {data.employment_len} years
- **Loan Purpose:** {data.loan_intent}
- **Requested Loan Amount:** ${data.loan_amnt}
- **Interest Rate:** {data.loan_int_rate}%  
- **Loan-to-Income Ratio:** {data.loan_percent_income}%  
- **Credit History Length:** {data.cred_hist_len} years  

### Loan Assessment Summary:
Based on the applicant’s profile, the backend evaluation system has provided the following probabilities:  
- **Probability of Loan Approval:** {prob_eligible:.2f}  
- **Probability of Loan Rejection:** {prob_not_eligible:.2f}  

---

### Your Task:

**For Eligible Applicants (High Probability of Approval)**  
- If the applicant has a **high probability of loan approval**, begin the response with:  
  **"Congratulations! Based on your financial profile, you are highly likely to receive loan approval."**  
- Summarize the **key strengths** that contributed to their eligibility (e.g., stable income, good credit history, low debt-to-income ratio).  
- Offer a **few concise financial tips** to maintain or further improve their loan prospects.  

**For Ineligible Applicants (Low Probability of Approval)**  
- If the applicant has a **low probability of loan approval**, begin with:  
  **"Based on the financial assessment, you may face challenges in securing loan approval. However, there are actionable steps you can take to improve your eligibility."**  
- Provide a **detailed breakdown of key factors** that contributed to rejection, such as:  
  - **Insufficient income** → Suggest ways to increase income (e.g., seeking higher-paying employment, additional income sources).  
  - **High debt-to-income ratio** → Explain strategies to reduce existing debt before applying again.  
  - **Short credit history or poor credit score** → Provide detailed guidance on building a stronger credit profile (e.g., timely bill payments, responsible credit usage, credit-building loans).  
  - **Unstable employment** → Advise on maintaining a consistent work record before reapplying.  
  - **High loan amount relative to income** → Recommend adjusting loan requests or exploring alternative loan options.  

### Additional Recommendations:
- If applicable, suggest **alternative financial solutions**, such as seeking a **co-signer**, applying for a **secured loan**, or considering government-backed loan programs.  
- Offer a **timeline for reapplication**, suggesting when they should reapply based on their financial improvement efforts.  

### Important Guidelines:
**DO NOT mention "model predictions" or "ML-based evaluation."**  
**Make the response sound like an expert financial advisor’s personalized assessment.**  
**Use a professional, structured, and empathetic tone.**  
**Ensure a concise response for eligible applicants and a highly detailed improvement plan for ineligible ones.**  
"""



    ans = llm.invoke(prompt)
    return {"message": ans.content}

# Run FastAPI server
configure()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
