import streamlit as st
from streamlit_option_menu import option_menu
import csv_agent
import Grievance_chatbot
 
# Set the page configuration
st.set_page_config(page_title="RFP Analyser")
 
# Apply subtle and minimal CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
    }
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 16px;
        margin: 0.5rem 0;
        cursor: pointer;
        font-family: 'Arial', sans-serif;
    }
    .stButton > button:hover {
        background-color: #005bb5;
    }
    .title {
        font-size: 2rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.25rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .nav-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
    }
    .nav-button:hover {
        background-color: #45a049;
    }
    .icon {
        color: #0066cc;
    }
    .icon:hover {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
 
def main():
    col1, col2 = st.columns([1, 5])  # Adjust width ratio as needed

    with col1:
        st.image("EY-Logo.png", width=85)  # Adjust width as needed

    with col2:
        st.markdown(
        """
        <h1 style="font-family: 'Calibri', sans-serif; font-size: 35px; color: #FFFFFF; margin: 0;">
        Grievance Management System
        </h1>
        """,
        unsafe_allow_html=True
    )
 
 
 
    # Using option menu for navigation
    selected = option_menu(
        menu_title=None,  # Required
        options=[ "Grievance Chatbot", "CSV Agent"],  # Required
        icons=["cpu", "robot"],  # Icons for pages
        menu_icon="cast",  # Optional
        default_index=0,  # Optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f9f9f9"},
            "icon": {"color": "#2e2e38", "font-size": "25px"},
            "nav-link": {"font-size": "20px", "text-align": "center","color": "#2e2e38", "margin": "0px", "--hover-color": "#eee", "font-family": "Calibri, sans-serif"},
            "nav-link-selected": {"background-color": "#FFE600", "color": "#2e2e38", "font-family": "Calibri, sans-serif"},
            "nav-icon-selected": {"color": "#2e2e38"},
        }
    )
 
    if selected == "Grievance Chatbot":
        Grievance_chatbot.run()
    elif selected == "CSV Agent":
        csv_agent.run()
 
if __name__ == "__main__":
    main()