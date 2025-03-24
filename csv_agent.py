import os
from pathlib import Path
from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from phi.model.azure import AzureOpenAIChat
from langchain_openai import AzureChatOpenAI
import streamlit as st
import numpy as np
import torch
import io
import soundfile as sf
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

model = AzureChatOpenAI(
    azure_endpoint="------",
    openai_api_version="--------",
    deployment_name="-------",
    openai_api_key="-----------",
    openai_api_type="azure",
    temperature=0
)

llm = AzureOpenAIChat(
    id="ey_id",  # Add this required field
    azure_endpoint="--------",
    api_version="-------",
    azure_deployment="-------",
    api_key="-----",
    api_type="azure",
    temperature=0
)

# Set up the base directory for temporary files
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

local_csv_path = "Grievance_Dataset.csv"

python_agent = PythonAgent(
    model=llm,  
    base_dir=tmp,  # Temporary directory for intermediate files
    files=[
        CsvFile(
            path=local_csv_path, 
            description="""This Dataset contains information about the grievances logged by multiple user.
            It has 8 columns:
            - **Grievance_Description**: The complaint details.
            - **Date**: YYYY-MM-DD format.
            - **Category**: Primary category of the grievance.
            - **Sub-Category**: Secondary category of the grievance.
            - **State**: Name of the state.
            - **District**: Name of the district.
            - **Address**: Full address of the grievance location.
            - **Priority**: Urgency of grievance (High, Medium, Low).

            Description of the columns
            1. Grievance Description: It contains the Grievances logged by the user.
            2. Date: Date of logging of Grievance (yyyy-mm-dd)
            3. Category: Name of the Primary Category of the Grievance. There are 7 Primary Categories in which Grievances are categorised.
                - Administrative & Public Services Issues
                - Law, Order & Human Rights Concerns
                - Telecom, Electricity & Water supply Complaints
                - Healthcare & Social Welfare Issues
                - Education & Student-Related Complaints  
                - Public Infrastructure & Civic Issues  
                - Environmental & Sanitation Concerns
       
            4. Sub-Category: Secondary Categories of the Grievance. There are 20 Secondary Categories in which Grievances are categorised.
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
                19. Punlic Infrastructure Issues
                20. Pollution & Environmental Hazards

            5. State: Name of the State of the area for which Grievance is registered.
            6. District: Name of the district of the area for which Grievance is registered.	
            7. Address: Complete address of the area for which Grievance is registered.
            8. Priority: Defines the urgency of the grievance categorised as high, medium, low.

            """,
        )
    ],
    markdown=True,  # Enable markdown formatting for responses
    pip_install=True,  # Install required dependencies automatically
    show_tool_calls=False,  # Display tool calls for better transparency
    instructions=[
        "First, get the list of available files.",
        "Then, check the column names and structure.",
        "If filtering is needed (like based on date, state, or category), apply the necessary filters.",
        "When the user asks for some informations in which a dataframe is returned it must contain grievances registered."
        "Then, execute the query on the filtered dataset.",
        "When asked to plot a graph save the image of the graph and just return the path of image saved nothing else."
        "Ensure the response is concise, formatted correctly, and directly answers the question. If a graph is to be plotted return Path:'Plath_of_image'"
    ],
    allow_code_execution=True
)

def contains_plot(text):
    return "Plot" in text

def get_path(content: str) -> str:
    prompt = (
            "You are AI- assistant you are gonna get a content of response by a agent which contains a path of an image. Your job is return that path of the image from the response.Give the response path as plain text not markdown."
            f"{content}"
            "Return only the path of the image without any additional text or explanations."
        )
    corrected_grievance = model.predict(prompt).strip()
    return corrected_grievance

def transcribe_audio(audio_file):
    """Converts recorded audio to text using Whisper ASR model."""
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio_file.getvalue())
    audio_data, sample_rate = sf.read("recorded_audio.wav")
    # Transcribe the audio
    transcription = pipe({"raw": audio_data, "sampling_rate": sample_rate}, generate_kwargs={"language": "english"})
    return transcription["text"]

def run():

    st.markdown(
    """
    <h2 style="font-family:'Calibri', sans-serif; font-size: 30px; color: #FFFFFF; margin: 0; text-align: center;">
    CSV Agent
    </h2>
    """,
    unsafe_allow_html=True
)
        
    with st.expander("DESCRIPTION", expanded=False):
        st.write("""
        The CSV Agent enables users to interact with logged grievance data, providing insights and answers for better understanding and analysis.
    """)

    st.markdown(
    """
    <h2 style="font-family: 'Calibri', sans-serif; font-size: 25px; text-align: center; color: #FFFFFF;">
        Interact with the grievance records to gain insights.
    </h2>
    """,
    unsafe_allow_html=True
    )

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
    </style>
    """,
    unsafe_allow_html=True
    )

    # Define session state for question input
    if "question" not in st.session_state:
        st.session_state.question = ""

    col1, col2 = st.columns([8, 2])  # Layout for text area and mic button

    with col1:
        # Populate the text input with the transcribed question if available
        question = st.text_input(
            "", placeholder="Ask me a question!",
            value=st.session_state.question or ""
        )

    with col2:
        audio_file = st.audio_input("Record", key="audio_input", label_visibility="collapsed")  # Hidden label for better UI

    # Process voice input
    if audio_file and not st.session_state.get("transcribedloc", False):
        st.write("Transcribing Audio...")
        st.session_state.question = transcribe_audio(audio_file)  # Transcribe the audio
        st.session_state.transcribedloc = True  # Mark as transcribed
        st.rerun()  # Refresh the app to update the text input field

    if st.button("Run Flow"):
        st.session_state.question = question
        if not st.session_state.question.strip():
            st.error("Please enter a valid question.")
        else:
            with st.spinner("Processing, give me a minute!"):  # Show a loading spinner
                response = python_agent.run(st.session_state.question or question)
                if contains_plot(st.session_state.question):
                    path = get_path(response.content)
                    st.image(path)
                else:
                    st.markdown(response.content) 