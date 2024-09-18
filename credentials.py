import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import os

def load_google_api_key() -> str:

    # GOOGLE_API_KEY = "AIzaSyC4dc4KNduUb3vWE5P-Ul87ejm3UY8XpO0"
    OPENAI_API_KEY = "sk-proj-m7hjRWIxvY7TQi0hASc3T3BlbkFJQLZ4m9HSCRpTxeXitwiN"

    return OPENAI_API_KEY

def load_yaml():
    # Load credentials
    with open("./cred.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)

    return config

def save_yaml(config):
    # Update the YAML file
    with open('./cred.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def load_authenticator(config):
    authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'])

    return authenticator
