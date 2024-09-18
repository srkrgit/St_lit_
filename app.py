import streamlit as st
import hydralit_components as hc
import streamlit_authenticator as stauth
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Self-Modules
from credentials import load_yaml, load_authenticator, save_yaml, load_google_api_key
from data_preprocess import DataPreprocess
from main import MainApp
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


#---------------------------------------------(Configuration)---------------------------------------------

st.set_page_config(layout = 'wide') 

config = load_yaml() # stores user credentials
authenticator = load_authenticator(config)

# os.environ["GOOGLE_API_KEY"] = load_google_api_key()
os.environ["OPENAI_API_KEY"] = load_google_api_key()

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",  transport="rest")
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

# llm = ChatGoogleGenerativeAI(model="gemini-pro",
#                  temperature=0.5)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2
)

#---------------------------------------------(Register User and Login Section)---------------------------------------------
 
# Define the layout using columns
col1, col2 = st.columns(2)

# Check if the user is logged in
if not st.session_state.get("authentication_status"):
    # Register form
    with col1:
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)

# Login form
with col2:
    authenticator.login()
    if st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')

# Hide the register form after successful login
if st.session_state["authentication_status"]:

    st.empty()
    
    with hc.HyLoader('', hc.Loaders.standard_loaders, index=[0]):

        #------------------(MAIN FUNCTION CONTAINS- data loading, data preprocessing, data fetching and LLM call)------------------
    
        if __name__ == "__main__":
            MainApp(embeddings, llm).run()

    with st.sidebar:
        authenticator.logout()
        st.markdown(f'Welcome ***{st.session_state["name"]}***')


save_yaml(config) # Save new user credentials

