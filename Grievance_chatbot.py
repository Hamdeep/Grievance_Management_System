from langchain.chat_models import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
import json
from langchain_openai import AzureChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import streamlit as st
import numpy as np
import torch
import pandas as pd
import soundfile as sf
from transformers import pipeline
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

genai.configure(api_key="API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"PATH"


# Initialize LLM before defining tools
llm = AzureChatOpenAI(
    azure_endpoint="------",
    openai_api_version="--------",
    deployment_name="-------",
    openai_api_key="-----------",
    openai_api_type="azure",
    temperature=0
)

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_audio(audio_file):
    """Converts recorded audio to text using Whisper ASR model."""
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio_file.getvalue())

    audio_data, sample_rate = sf.read("recorded_audio.wav")
    
    # Transcribe the audio
    transcription = pipe({"raw": audio_data, "sampling_rate": sample_rate}, generate_kwargs={"language": "english"})
    
    return transcription["text"]

# Define tools for extracting information
def log_grievance(grievance: str) -> str:
    """Corrects spelling and grammatical errors in the given grievance without altering its meaning or adding extra details."""
    prompt = (
        "Correct any spelling or grammatical mistakes in the following sentence while keeping it natural and unchanged in tone:\n\n"
        f"{grievance}"
        "Return only the corrected sentence without any additional text or explanations."
    )
    corrected_grievance = llm.predict(prompt).strip()
    return corrected_grievance

def assess_urgency(category: str) -> str:
    high=["Police Misconduct or Delayed Action","Human Rights Violations","Harassment in Educational Institutions","Domestic Violence and Women’s Safety","Poor Medical Treatment, Hospital Negligence","Water Supply Issue","Road Maintenance and Public Transport","Legal Aid & Judicial Delays","Electricity Power Outages","Pollution & Environmental Hazards"]
    med=["Corruption or Misconduct in Government Offices","Delayed Government Services","Electricity Billing and Power Outage Complaints","Waste Management & Garbage Disposal Issues","Consumer Rights Violations","Issues with Government Health Schemes (Ayushman Bharat, ESIC)","Admissions Disputes in Schools & Colleges","Public Infrastructure Issues"]
    low=["Broadband and Mobile Service Complaints","Problems in Pension Payments","Issues with Scholarships, Financial Aid"]

    lists = {"High": high, "Medium": med, "Low": low}

    for name, lst in lists.items():
        if category in lst:
            return name
            break

def assess_severity(urgency: str) -> str:
    """Uses LLM to determine the urgency level based on user input and ensures the output is High, Medium, or Low."""
    prompt = ("""
    Role & Task:
    You are an AI assistant in a Grievance Management System. Your task is to analyze the .

    ## Severity Classification Criteria:
    ### High Severity (Immediate attention required, serious consequences):
    Grievances that pose life-threatening risks or could lead to serious future consequences.

    Examples:
    Violent crimes or human rights violations.
    Water shortages affecting large populations.
    Life-threatening situations in hospitals (e.g., lack of oxygen or critical supplies).
    Wrongful imprisonment or denial of justice.
    Electricity power outages impacting hospitals or emergency services.
    Pollution or environmental hazards causing severe health risks.
    Classification Rule: If the grievance matches the above examples or is of similar importance, classify it as High.

    ### Medium Severity (Needs attention but not life-threatening):
    Grievances that require attention but do not pose immediate life-threatening risks.

    Examples:
    Corruption or misconduct in government offices. Eg: The regional officer is asking for a bribe to issue a land ownership certificate."
    Delays in government services (e.g., document processing). Eg. PAN card application stuck for months.
    Mobile service issues or billing errors.   Eg. Frequent internet disconnection in my area.
    Water supply issues (non-critical).     Eg. Irregular water supplies in area.
    Delays in pensions or admission disputes.   Eg. Pension approval delayed despite repeated visits.
    Roads and infrastructure-related complaints.    Eg. Potholes on main road causing inconvenience.
    Waste management issues.

    Classification Rule: If the grievance matches the above examples or is of similar importance, classify it as Medium.

    ### Low Severity (Minor inconvenience, can be resolved without urgency):
    Grievances that describe minor inconveniences or issues that can be resolved without urgency.

    Examples:
    Minor footpath or streetlight issues.
    Customer rights issues like minor overcharging.
    Non-critical complaints about public amenities.
    Classification Rule: If the grievance matches the above examples or is of similar importance, classify it as Low.

    ## Output Instructions:
    Strictly return only one word: "High", "Medium", or "Low" based on the classification above.
    Do not provide explanations, additional text, or any other information.
              
    Input:
        Urgency description: '{urgency}'
"""
    )
    response = llm.predict(prompt).strip()
    return response  # Return only the processed urgency level

def assess_state(address: str) -> str:
    prompt = (
        f"""Extract and return the Indian state name from the given address. If the state name has a spelling mistake, correct it. If no valid Indian state is mentioned, return 'Nope'. Do not infer or assume the state from city name or by the adress if it is missing. Output only the corrected state name or 'Nope'—no extra words or explanations.
        Location Address: "{address}"
        """
    )
    response = llm.predict(prompt).strip()
    return response  # Return only the processed urgency level


def assess_dist(address: str, state: str) -> str:
    prompt = (
    f"""You are an expert in Indian geography. Extract and return the correct Indian district name from the given address. 
    If the district name has a minor spelling mistake, correct it. 
    If the mentioned district does not belong to the specified state or is missing, return 'Nope'. 
    Ensure that you validate the district against officially recognized district names of India. 
    Do not infer or assume the district name from the city name or any other part of the address.
    
    Example Inputs & Outputs:
    - Address: "Patna", State: "Bihar" → Output: "Patna"
    - Address: "Patiala", State: "Punjab" → Output: "Patiala"
    - Address: "Mumbai", State: "Punjab" → Output: "Nope"
    - Address: "Patan", State: "Bihar" → Output: "Nope"

    Location Address: "{address}"  
    State: "{state}"  

    Output only the corrected and valid Indian district name or 'Nope'—no extra words or explanations.
    """
)
    response = llm.predict(prompt).strip()   
    return response  # Return only the processed urgency level

@tool
def categorize_grievance(grievance: str) -> str:
    """Uses LLM to determine the most relevant category for the given grievance."""
    prompt = f"""
    You are an AI assisting in a Grievance Management System. Categorize the given grievance into one of the predefined categories below:

    Categories:
    1. Corruption or Misconduct in Government Offices
    2. Delayed Government Services
    3. Harassment in Educational Institutions 
    4. Electricity Billing and Power Outage Complaints 
    5. Road Maintenance and Public Transport  
    6. Waste Management & Garbage Disposal Issues 
    7. Police Misconduct or Delayed Action 
    8. Legal Aid & Judicial Delays 
    9. Domestic Violence and Women’s Safety  
    10. Human Rights Violations
    11. Consumer Rights Violations
    12. Broadband and Mobile Service Complaints
    13. Water Supply Issue
    14. Poor Medical Treatment, Hospital Negligence
    15. Issues with Government Health Schemes (Ayushman Bharat, ESIC)
    16. Problems in Pension Payments
    17. Issues with Scholarships, Financial Aid
    18. Admissions Disputes in Schools & Colleges
    19. Public Infrastructure Issues
    20. Pollution & Environmental Hazards
    21. Spam (if the grievance is irrelevant, random, or spam)

    Grievance: "{grievance}"

    Strictly return only the category name from the list above, nothing else.
    """
    category = llm.predict(prompt).strip()
    return category


# agent = initialize_agent(
#     tools=[log_grievance, assess_severity],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     memory=memory
# )

def get_department(user_input,category):
    prompt = f"""
    You are an AI assisting in a Grievance Management System. Categorize the given grievance into one of the predefined departments below, Each department contains these categories. Use the category to determine which department it belongs to:

    Departments and categories under them:
    1. Administrative & Public Services Issues
        a. Delayed Government Services 
        b. Corruption or Misconduct in Government Offices
    2. Law, Order & Human Rights Concerns
        a. Police Misconduct or Delayed Action
        b. Human Rights Violations
        c. Legal Aid & Judicial Delays 
        d. Domestic Violence and Women’s Safety
        e. Consumer Rights Violations
    3. Telecom, Electricity & Water supply Complaints
        a. Broadband and Mobile Service Complaints
        b. Electricity Billing and Power Outage Complaints
        c. Water Supply Issue 
    4. Healthcare & Social Welfare Issues
        a. Poor Medical Treatment, Hospital Negligence
        b. Issues with Government Health Schemes (Ayushman Bharat, ESIC)
        c. Problems in Pension Payments (Old Age, Widow, etc)  
    5. Education & Student-Related Complaints
        a. Admissions Disputes in Schools & Colleges
        b. Issues with Scholarships, Financial Aid
        c. Harassment in Educational Institutions
    6. Public Infrastructure & Civic Issues
        a. Road Maintenance and Public Transport  
        b. Infrastructure Issues
    7. Environmental & Sanitation Concerns
        a. Waste Management & Garbage Disposal Issues
        b. Pollution & Environmental Hazards
    8. Spam (If the category is Spam)

    Grievance: "{user_input}"
    Category: "{category}"

    Strictly return only the Department name from the list above, nothing else.
    """
    category = llm.predict(prompt).strip()
    return category

def getChain():
    prompt_template = """
    ### Role  
    You are an AI assistant specialized in grievance management. Your task is to analyze grievances and provide **logical and relevant** resolutions based on historical data.  

    ### Context  
    - You will be given a **markdown-formatted dataset** containing two columns:  
    1. **Grievance Description** – A record of past grievances.  
    2. **Solution** – The resolution provided for each grievance.  
    - Your goal is to understand this dataset and use it as a reference to suggest solutions for newly logged grievances. The solution should be of admin prospective not for the user who submitted the grievance. 

    ### Objective  
    For each new grievance:  
    - **Analyze** its nature and context.  
    - **Compare** it with historical grievances and their solutions.  
    - **Generate** a well-reasoned, actionable, and relevant resolution based on past data.  
    - If no exact match is found, provide a **generalized solution** based on similar grievances while ensuring logical accuracy.  

    ### Requirements  
    - Ensure solutions are **practical, well-structured, and aligned** with real-world grievance resolution mechanisms.  
    - If a grievance has multiple possible solutions, suggest the **most effective** approach first.  
    - If historical data lacks a relevant resolution, state that explicitly and propose a **general best practice solution**.  

    ### Example Input Format (Markdown Table)  
    | Grievance Description | Solution |
    |-----------------------|----------|
    | "Frequent power outages in my locality disrupt daily life." | "Coordinate with the electricity board to assess power stability in the affected area. If the issue persists, initiate discussions with higher authorities for infrastructure improvements." |
    | "Garbage collection in my area is irregular, leading to unhygienic conditions." | "Assign sanitation teams to conduct regular waste collection in the affected area. Review the existing waste management schedule and make adjustments if necessary." |

    
    Expected Output
    For a new grievance "Streetlights in my neighborhood are not working, causing safety concerns at night.", the AI should generate:
    "Notify the municipal electricity department to inspect and repair faulty streetlights. Ensure follow-ups with the maintenance team until resolved. If budget constraints exist, escalate the matter for approval and prioritize repairs based on urgency."

    Ensure your solutions are clear, logical, and actionable, helping users resolve grievances efficiently.

    Note: The final response should always be just resolution dont add any extra words or sentences.
    Inputs:
    - Historic Data: {context}
    - Grievance : {question}
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | llm | StrOutputParser()
    return chain

def get_sol(user_question):
    embd = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db_name = "resolv"
    db = FAISS.load_local(db_name, embd, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=3)
    chain = getChain()
    response = chain.invoke({"context": docs, "question": user_question}, return_only_outputs=True)
    return response

def get_vectorstore(df):
    documents = []
    for _, row in df.iterrows():
        ddf = row.to_frame().T  
        mrk = df_to_markdown(ddf)  

        doc = Document(page_content=mrk)
        documents.append(doc)

    create_vectorStore(documents)

def df_to_markdown(df):
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"

    rows = "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.values
    )

    markdown_table = f"{header}\n{separator}\n{rows}"
    return markdown_table

def create_vectorStore(text):
    embd = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db_name = "resolv"
    vectorstore = FAISS.from_documents(text, embedding=embd)
    vectorstore.save_local(db_name)

def run():
    """Handles multi-turn grievance conversation in Streamlit."""
    
    st.markdown(
    """
    <h2 style="font-family:'Calibri', sans-serif; font-size: 30px; color: #FFFFFF; margin: 0; text-align: center;">
    AI-Powered Grievance Assistant
    </h2>
    """,
    unsafe_allow_html=True
)

    with st.expander("DESCRIPTION", expanded=False):
        st.write("""
            This chatbot assists users in registering grievances across various categories and sub-categories, including Telecom, Electricity & Water Supply complaints, Law & Order issues, and Human Rights concerns.
        """)

    st.markdown(
    """
    <style>
    div.stButton { 
        display: flex;
        justify-content: center;
    }
    div.stButton > button {
        background-color: #ffe600;
        color: black;
        font-size: 16px;
        font-weight: bold;
        border-radius: 6px;
        padding: 8px 16px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #ffe600;
        color: black;
        transform: scale(1.05);
    }
    .stAudioInput > div > button { 
            color: #000000 !important; /* Change to desired color */
        }
    .stAudioInput > div > span { 
            color: #000000 !important; /* Change to desired color */
            font-weight: bold;
        }
    .stAudioInput button {
            filter: invert(0%) brightness(0%) contrast(100%);
        }
    """,
    unsafe_allow_html=True
)
    
    st.markdown(
    """
    <h2 style="font-family: 'Calibri', sans-seri; font-size: 18px; text-align: center; color: #FFFFFF;">
        Submit your grievance below to initiate the process.
    </h2>
    """,
    unsafe_allow_html=True
    )
    
    # Initialize session state variables
    for key in ["messages", "state", "district", "urgency", "grievance_data", "initialized", "transcribed_grievance", "transcribed_location", "transcribed"]:
        if key not in st.session_state:
            st.session_state[key] = None if key not in ["messages", "transcribed_grievance", "transcribed_location", "transcribed"] else []

    if not st.session_state.initialized:
        st.session_state.messages.append({"role": "assistant", "content": "Hi! What is your current issue?", "color": "#4CAF50"})
        st.session_state.initialized = True

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Step 1: User enters Grievance
    if st.session_state.state is None:
        col1, col2 = st.columns([8, 2])

        with col1:
            grievance_text = st.text_input("Enter Your Grievance...", value=st.session_state.transcribed_grievance or "")

        with col2:
            audio_file = st.audio_input("Record Grievance", key="audio_grievance", label_visibility="collapsed") 

        if audio_file and not st.session_state.transcribed:
            st.session_state.transcribed_grievance = transcribe_audio(audio_file)  # Function to convert speech to text
            st.session_state.transcribed = True  # Mark as transcribed
            st.rerun()  # Refresh to update input field

        if st.button("Submit Grievance") and (grievance_text or st.session_state.transcribed_grievance):
            st.session_state.messages.append({"role": "user", "content": grievance_text or st.session_state.transcribed_grievance})
            st.session_state.transcribed = False  # Reset flag after submission
            corrected_grievance = log_grievance(grievance_text or st.session_state.transcribed_grievance)
            category = categorize_grievance(corrected_grievance)
            department_grievance = get_department(corrected_grievance, category)
            urgency = assess_urgency(category)

            vector_store_path = f"resolv.faiss"
            if not os.path.exists(vector_store_path):
                df=pd.read_csv("Refined_Grievance_Solutions.csv", encoding='ISO-8859-1')
                get_vectorstore(df)
                print("Db_created")

            resl=get_sol(corrected_grievance)

            if category == "Spam":
                with open("spam_logs.txt", "a") as file:
                    file.write(f"{datetime.now().strftime('%Y-%m-%d')} - {corrected_grievance}\n")
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ Spam identified. Grievance not logged."})
            else:
                st.session_state.grievance_data = {
                    "Grievance Description": corrected_grievance,
                    "Category": department_grievance,
                    "Sub-Category": category,
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Priority": urgency,
                    "Potential Resolution": resl
                }

                st.session_state.messages.append({"role": "assistant", "content": "Please enter your complete address."})
                st.session_state.state = "waiting_for_location"

            st.session_state.transcribed_grievance = ""  # Reset transcribed text
            st.rerun()

    # Step 2: User enters Location (State & District)
    if st.session_state.state == "waiting_for_location":
        col1, col2 = st.columns([8, 2])

        with col1:
            # Populate the text input with the transcribed location if available
            location = st.text_input(
                "Enter your address, along with State and District.",
                value=st.session_state.transcribed_location or ""
            )

        with col2:
            audio_file = st.audio_input("Record Location", key="audio_location", label_visibility="collapsed") 

        # Check if audio is uploaded and not yet transcribed
        if audio_file and not st.session_state.get("transcribedloc", False):
            st.write("Transcribing Audio...")
            st.session_state.transcribed_location = transcribe_audio(audio_file)  # Transcribe the audio
            st.session_state.transcribedloc = True  # Mark as transcribed
            st.rerun()  # Refresh to update the input field with the transcribed text

        # If the location is submitted (either via text or transcribed audio)
        if st.button("Submit Address") and (location or st.session_state.transcribed_location):
            st.session_state.messages.append({"role": "user", "content": location or st.session_state.transcribed_location})
            st.session_state.transcribedloc = False  # Reset the transcription flag
            state = assess_state(location or st.session_state.transcribed_location)

            if state == "Nope":
                st.session_state.messages.append({"role": "assistant", "content": "❌ Invalid state. Please re-enter the address."})
            else:
                district = assess_dist(location or st.session_state.transcribed_location, state)
                if district == "Nope":
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ District does not match {state}. Please re-enter the address."})
                else:
                    st.session_state.grievance_data["State"] = state
                    st.session_state.grievance_data["District"] = district
                    st.session_state.grievance_data["Address"] = location or st.session_state.transcribed_location

                    st.session_state.messages.append({"role": "assistant", "content": "Your grievance has been logged!"})
                    st.session_state.state = "waiting_for_urgency"  # Move to the next step

            st.session_state.transcribed_location = ""  # Reset transcribed text
            st.rerun()  # Refresh to update the UI

    # Step 3: User enters Urgency
    if st.session_state.state == "waiting_for_urgency":
      
        column_order = [
        "Grievance Description", "Date", "Category", "Sub-Category",
        "State", "District", "Address", "Priority", "Potential Resolution"
    ]
        new_row = pd.DataFrame([st.session_state.grievance_data])

        new_row = new_row[column_order]
        csv_file = "Grievance_Dataset.csv"

        file_exists = os.path.exists(csv_file)
        new_row.to_csv(csv_file, mode="a", header=not file_exists, index=False)

        st.session_state.state = None