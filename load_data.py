import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


#---------------------------------------------(DataLoader Class- Loads CSVs files and Vector store)---------------------------------------------

class DataLoader:

    """
    A class to manage data loading and caching in a Streamlit application.
    
    Attributes:
    ----------
    embeddings : object
        Embeddings used for FAISS indexing.
        
    Methods:
    --------
    load_data(file_path):
        Loads data from a CSV file and caches it using Streamlit's caching mechanism.
        
    load_files():
        Loads multiple CSV files into Streamlit session state, including embeddings with FAISS.
    """
    
    def __init__(self, embeddings):
        self.embeddings = embeddings

    @st.cache_data
    def load_data(_self, file_path):

        """
        Loads data from a CSV file and caches it using Streamlit's caching mechanism.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file to load.
        
        Returns:
        --------
        pandas.DataFrame
            Loaded DataFrame from the CSV file.
        """

        # Function to load data
        data = pd.read_csv(file_path, dtype={'column_name': str})

        return data

    def load_files(self):

        """
        Loads multiple CSV files into Streamlit session state, including embeddings with FAISS.
        
        Uses:
        -----
        - RN DASHBOARD FY24 - Sookshma and Laghu - YTD Mar.csv
        - Cross Sell Final as on Apr'24 with IIB.csv
        - Cross_Sell_1405.csv
        - FY22,23,24 Claims Data SME 1.csv
        - FAISS index from './faiss_index'
        """

        if 'RN_df' not in st.session_state:
            st.session_state.RN_df = self.load_data("../Data/Sample_Set_2mo/CSV/RN DASHBOARD FY24 - Sookshma and Laghu - YTD Mar.csv")

        if 'cross_sell_sn_df' not in st.session_state:
            st.session_state.cross_sell_sn_df = self.load_data("../Data/Sample_Set_2mo/CSV/Cross Sell Final as on Apr'24 with IIB.csv")

        if 'cross_sell_details_df' not in st.session_state:
            st.session_state.cross_sell_details_df = self.load_data("../Data/Sample_Set_2mo/CSV/Cross_Sell_1405.csv")

        if 'raw_claims_df' not in st.session_state:
            st.session_state.raw_claims_df = self.load_data('../Data/Sample_Set_2mo/CSV/FY22,23,24 Claims Data SME 1.csv')

        if 'new_db' not in st.session_state:
            st.session_state.new_db = FAISS.load_local("./faiss_index_openai", self.embeddings, allow_dangerous_deserialization=True)