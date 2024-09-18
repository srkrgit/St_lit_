import streamlit as st
from load_data import DataLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma, FAISS
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np


class DataPreprocess:

    """
    Initializes the DataPreprocess class with embeddings and llm.

    Parameters:
    -----------
    embeddings : object
        Embeddings used for preprocessing.
    llm : object
        Some other object needed for preprocessing.
    """


    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm

        DataLoader(self.embeddings).load_files()

    @st.cache_data
    def RN_X_dfs(_self):
       
        """
        Preprocesses and returns dataframes related to RN_Xsellsn, cross_sell_sn_df, and top3_frequent_combined for fetching Policy Details. 
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple containing processed dataframes for RN_Xsellsn, cross_sell_sn_df, and top3_frequent_combined.
        """

        RN_df = st.session_state.RN_df 
        cross_sell_sn_df = st.session_state.cross_sell_sn_df
        cross_sell_details_df = st.session_state.cross_sell_details_df 

        X_sell_df = cross_sell_sn_df.copy()

        X_sell_df = X_sell_df[['POLICY_NO', 'base_lob', 'CLIENTNAME', 'TIMES_RENEWED_CNT', 'PRODUCER_CD', 'NEW_LOB', 'CL_ZONE', 'FINAL_CHANNEL', 'MAIN_CHANNEL', 'SUM_INSURED', 
                            'AIGC', 'AIGC-EC', 'CASUALTY', 'ENGG.', 'FL', 'GHI', 'GPA', 'MARINE-NON SMP', 'MARINE-SMP', 'AIGC_Prem', 'AIGC-EC_Prem',
                                'CASUALTY_Prem', 'ENGG._Prem', 'FL_Prem', 'GHI_Prem', 'GPA_Prem', 'MARINE-NON SMP_Prem', 'MARINE-SMP_Prem', 'occupancy_name',
                                'PREMIUMAMOUNT']]

        X_sell_df[X_sell_df['base_lob'] == 'AIGC'].drop(columns=['base_lob'])

        columns_to_fill = ['AIGC-EC', 'CASUALTY', 'ENGG.', 'FL', 'GHI', 'GPA', 'MARINE-NON SMP']

        X_sell_df[columns_to_fill] = X_sell_df[columns_to_fill].fillna(0).astype(int)

        # Create 'cross_sold' column
        X_sell_df['cross_sold'] = X_sell_df[columns_to_fill].max(axis=1).apply(lambda x: 1 if x > 0 else 0)

        # Filter rows where 'cross_sold' is 1 and drop the column
        undr_smpld_df = X_sell_df[X_sell_df['cross_sold'] == 1].drop(columns=['cross_sold']).reset_index(drop=True)

        # Group by 'occupancy_name' and sum the occurrences
        grouped = undr_smpld_df.groupby('occupancy_name')[columns_to_fill].sum()

        # Get the top 3 most frequent columns and their counts for each 'occupancy_name'
        top3_frequent_combined = grouped.apply(
            lambda row: pd.Series({
                'most_frequent_columns': row.nlargest(3).index.tolist(),
                'counts': row.nlargest(3).values.tolist()
            }), axis=1).reset_index()

        RN_df = RN_df.dropna(subset=['Producer Code', 'OLD_POLICY_NO'])
        RN_df.loc[:, 'Producer Code'] = RN_df['Producer Code'].astype(int)
        RN_df['OLD_POLICY_NO'] = RN_df['OLD_POLICY_NO'].astype(int)

        RN_Xsellsn = pd.merge(RN_df, cross_sell_sn_df, left_on='OLD_POLICY_NO', right_on='POLICY_NO').drop(['OLD_POLICY_NO'],axis=1).reset_index(drop=True)
        
        RN_Xsellsn = RN_Xsellsn[RN_Xsellsn.base_lob == 'AIGC']
        RN_Xsellsn = RN_Xsellsn[RN_Xsellsn.CL_ZONE.isin(['MUMBAI', 'PUNE'])]

        cross_sell_details_df = cross_sell_details_df.dropna(subset='POLICY_NO')

        cross_sell_details_df['POLICY_NO'] = cross_sell_details_df['POLICY_NO'].astype(int)

        return RN_Xsellsn, cross_sell_sn_df, top3_frequent_combined


    def prod_cust(self, producer_cd : int, pol_num: int):

        """
        Processes customer data based on producer code and policy number for fetching Policy Details.

        Parameters:
        -----------
        producer_cd : int
            Producer code for filtering.
        pol_num : int
            Policy number for filtering.

        Returns:
        --------
        pd.DataFrame.styler
            Styled DataFrame showing processed customer data.
        """

        product_name_mapping_dict = {'BUSINESSGUARDSOOKSHMAUDYAMSURAKSHA': 'Business Guard Sookshma Udyam Suraksha',
        'BUSINESSGUARDLAGHUUDYAMSURAKSHA': 'Business Guard Laghu Udyam Suraksha',
        'OPENPOLICYMARINECARGO': 'Open Policy Marine Cargo',
        'AIGCOMBINEDPACKAGEPOLICY': 'AIG Combined Package Policy',
        'MARINETEACROPPOLICY': 'Marine Tea Crop Policy',
        'EMPLOYEECOMPENSATION': 'Employee Compensation',
        'CPMINSURANCE': 'CPM Insurance',
        'MARINESPECIFICPOLICY': 'Marine Specific Policy',
        'PROJECTINSURANCE': 'Project Insurance',
        'PUBLICLIABILITYACTS': 'Public Liability Acts',
        'GROUPACCIDENTGUARD': 'Group Accident Guard',
        'GROUPMEDICAREPOLICY': 'Group Medicare Policy',
        'COMMERICALGENERALLIABILITY': 'Commerical General Liability',
        'DIRECTORANDOFFICERLIABILITY': 'Director And Officer Liability',
        'BHARATGRIHARAKSHAAIGC': 'Bharat Griha Raksha AIGC',
        'PUBLICLIABILITYINS': 'Public Liability Ins',
        'PUBLICLIABILITYPRODUCT': 'Public Liability Product'}

        RN_Xsellsn, cross_sell_sn_df, top3_frequent_combined = self.RN_X_dfs()

        filtered_RN_Xsellsn = RN_Xsellsn[RN_Xsellsn.PRODUCER_CD == int(producer_cd)]

        customer_filtered = filtered_RN_Xsellsn[filtered_RN_Xsellsn.POLICY_NO == int(pol_num)]

        customer_filtered = customer_filtered[['POLICY_NO', 'standard_name', 'PRODUCT_NAME', 'POL_EXP_DATE', 'PRODUCERNAME', 'PRODUCER_CD', 'occupancy_name',
                            'TIMES_RENEWED_CNT', 'NEW_LOB', 'BUSINESS_TYPE', 'CL_ZONE', 'PREMIUMAMOUNT', 'SUM_INSURED', 'SI_Cat', 'Prem_Cat', 'AIGC & <50L SI']]
        
        customer_filtered['PRODUCT_NAME'] = product_name_mapping_dict[customer_filtered['PRODUCT_NAME'].iloc[0]]
        customer_filtered['PREMIUMAMOUNT'] = f"{customer_filtered['PREMIUMAMOUNT'].iloc[0]*10e4:.2f}K"
        customer_filtered['SUM_INSURED'] = f"{customer_filtered['SUM_INSURED'].iloc[0]:.2f} Cr"

        SI_cat_LOB_prem = {}

        cat_list = list(cross_sell_sn_df[cross_sell_sn_df.occupancy_name == str(customer_filtered.occupancy_name.iloc[0])]['SI_Cat'].unique())

        for cat in cat_list:
            pol_count_dict = cross_sell_sn_df[cross_sell_sn_df['SI_Cat'] == str(cat)][cross_sell_sn_df[cross_sell_sn_df['SI_Cat'] == str(cat)].occupancy_name == str(customer_filtered.occupancy_name.iloc[0])][['AIGC', 'AIGC-EC', 'CASUALTY', 'ENGG.', 'FL', 'GHI', 'GPA', 'MARINE-NON SMP', 'MARINE-SMP']].sum().to_dict()
            lob_occu_prem_dict = cross_sell_sn_df[cross_sell_sn_df['SI_Cat'] == str(cat)][cross_sell_sn_df[cross_sell_sn_df['SI_Cat'] == str(cat)].occupancy_name == str(customer_filtered.occupancy_name.iloc[0])][['AIGC_Prem', 'AIGC-EC_Prem', 'CASUALTY_Prem', 'ENGG._Prem', 'FL_Prem', 'GHI_Prem', 'GPA_Prem', 'MARINE-NON SMP_Prem', 'MARINE-SMP_Prem']].sum().to_dict()
            
            result_dict = {
                k: (lob_occu_prem_dict[k] / pol_count_dict[k.replace('_Prem', '')] if pol_count_dict[k.replace('_Prem', '')] != 0 else 0.0) 
                for k in lob_occu_prem_dict.keys()
                }
            
            SI_cat_LOB_prem[cat] = result_dict

        prem_dict = {}

        for key in top3_frequent_combined[top3_frequent_combined.occupancy_name == customer_filtered.occupancy_name.iloc[0]]['most_frequent_columns'].iloc[0]:
            key = f'{key}_Prem'
            if key in SI_cat_LOB_prem[customer_filtered.SI_Cat.iloc[0]].keys():
                premium = f"{SI_cat_LOB_prem[customer_filtered.SI_Cat.iloc[0]][key] * 10e4:.2f}K"
                prem_dict[key] = premium
        
        df = pd.DataFrame([{'POLICY_NO': 'NA',
                            'standard_name': customer_filtered.standard_name.iloc[0], 
                            'PRODUCT_NAME': 'NA', 
                            'POL_EXP_DATE': 'NA',
                            'PRODUCERNAME': customer_filtered.PRODUCERNAME.iloc[0], 
                            'PRODUCER_CD': customer_filtered.PRODUCER_CD.iloc[0], 
                            'occupancy_name': customer_filtered.occupancy_name.iloc[0], 
                            'TIMES_RENEWED_CNT': 'NA',
                            'NEW_LOB': cross_lob[:-5], 
                            'BUSINESS_TYPE': 'NA', 
                            'CL_ZONE': customer_filtered.CL_ZONE.iloc[0], 
                            'PREMIUMAMOUNT': prem_dict[cross_lob], 
                            'SUM_INSURED': customer_filtered.SUM_INSURED.iloc[0],
                            'SI_Cat': customer_filtered.SI_Cat.iloc[0], 
                            'Prem_Cat': customer_filtered.Prem_Cat.iloc[0],
                            'AIGC & <50L SI': customer_filtered['AIGC & <50L SI'].iloc[0]} for cross_lob in prem_dict])
                            
        customer_filtered = pd.concat([customer_filtered, df], ignore_index= True)
        
        customer_filtered.columns = ['Policy Number', 'Standard Name', 'Product Name', 'Policy Expiry Date',
                                     'Producer Name', 'Producer CD', 'Occupancy Name', 'Times Renewed Count', 
                                     'LOB', 'Business Type', 'CL Zone', 'Premium Amount', 'Sum Insured', 'SI Category',
                                     'Premium Category', 'AIGC & <50L SI']

        new_index = ['Renewal'] + ['Cross_Sell_' + str(i) for i in range(1, len(customer_filtered))]
        customer_filtered.index = new_index

        def color_col(x):
            cs = 'background-color: #774c60'
            rn = 'background-color: #372549'
            highlight = 'background-color: #b75d69'

            df_1 = pd.DataFrame('', index=x.index, columns=x.columns)
            df_1.iloc[:, 1:] = cs
            df_1.loc[['LOB', 'Premium Amount'], 'Cross_Sell_1':] = highlight
            df_1.iloc[:, :1] = rn

            return df_1

        cell_hover = {
            "selector": "td:hover",
            "props": [("background-color", "#1E90FF")]
        }

        index_names = {
            "selector": ".index_name",
            "props": "font-style: italic; color: darkgrey; font-weight:normal;"
        }
        headers = {
            "selector": "th:not(.index_name)",
            "props": "background-color: #1a1423; color: white;"
        }
        return customer_filtered.T.style.apply(color_col, axis=None).set_table_styles([cell_hover, index_names, headers]).set_properties(color="#eacdc2")


    @st.cache_data
    def load_claims_df(_self):

        """
        Loads and filters claims data from session state for Claims Details.

        Returns:
        --------
        pd.DataFrame
            Filtered and processed claims DataFrame.
        """

        claims_df =  st.session_state.raw_claims_df.copy()
        claims_df = claims_df[['Claim_Number', 'Policy_No', 'PRODUCER_CD', 'Insured_Name', 'Claimant_Name', 'Product_Name_As_Per_Register', 'Loss_Date', 'Pol_Incept_Date', 'Month',
                        'Cause_Of_Loss', ' NLI ',  'GLI', 'Branch Type_Comm',  'City_Name', 'New_Zone', 'State', 'Producer_Name', 'Revised_SME']]

        claims_df = claims_df[(claims_df['Branch Type_Comm'] == 'top 15') & 
                (claims_df['State'] == 'maharashtra') & 
                (claims_df['City_Name'].isin(['mumbai', 'pune'])) &
                (claims_df['Revised_SME'] == 'SME')]

        # Convert the columns to integers

        claims_df['Policy_No'] = claims_df['Policy_No'].astype(int)
        claims_df['PRODUCER_CD'] = claims_df['PRODUCER_CD'].astype(int)
        claims_df.rename(columns={'Policy_No': 'POLICY_NUMBER'}, inplace= True)
        claims_df.dropna(subset=['Pol_Incept_Date'], inplace = True)

        claims_df['GLI'] = claims_df['GLI'].astype(float).astype(int)

        claims_df = claims_df.reset_index(drop=True)

        return claims_df    

    def claims_info(self, pol_num: int):

        """
        Retrieves and formats claims information for a given policy number for Claims Details.

        Parameters:
        -----------
        pol_num : int
            Policy number to retrieve claims information for.

        Returns:
        --------
        pd.DataFrame
            Formatted DataFrame containing claims information.
        """

        cust_claims_df = self.load_claims_df()

        try:
            cause_of_loss_list = cust_claims_df[cust_claims_df['POLICY_NUMBER'] == pol_num]['Cause_Of_Loss'].unique()
            claim_info = pd.DataFrame({
                'Policy Number': pol_num,
                'Insured_Name': cust_claims_df[cust_claims_df['POLICY_NUMBER'] == pol_num]['Insured_Name'].unique()[0],
                'Cause of Losses': list(cust_claims_df[cust_claims_df['POLICY_NUMBER'] == pol_num]['Cause_Of_Loss'].unique()),
                'Product Name' : str(cust_claims_df[cust_claims_df['POLICY_NUMBER'] == pol_num]['Product_Name_As_Per_Register'].unique()[0]),
                'GLI_COL': f"{[cust_claims_df[(cust_claims_df['POLICY_NUMBER'] == pol_num) & (cust_claims_df['Cause_Of_Loss'] == col)]['GLI'].sum() * 10**-3 for col in cause_of_loss_list][0]:.2f}K",
                'Claims Count' : [cust_claims_df[(cust_claims_df['POLICY_NUMBER'] == pol_num) & (cust_claims_df['Cause_Of_Loss'] == col)]['Claim_Number'].nunique() for col in cause_of_loss_list]
                })
        except:
            claim_info = pd.DataFrame({
                'Policy Number': pol_num,
                'Insured_Name': 'NA',
                'Cause of Losses': 'NA',
                'Product Name' : 'NA',
                'GLI_COL': 'NA',
                'Claims Count' : [0]
                })

        def color_col(x):
            claim = 'background-color: #7678ed'

            df_1 = pd.DataFrame('', index=x.index, columns=x.columns)
            df_1.iloc[:, :] = claim

            return df_1

        cell_hover = {
            "selector": "td:hover",
            "props": [("background-color", "#1E90FF")]
        }

        index_names = {
            "selector": ".index_name",
            "props": "font-style: italic; color: darkgrey; font-weight:normal;"
        }
        headers = {
            "selector": "th:not(.index_name)",
            "props": "background-color: #3d348b; color: white;"
        }

        return claim_info.T.style.apply(color_col, axis=None).set_table_styles([cell_hover, index_names, headers]).set_properties(color="#ffffff")
    

    @st.cache_data
    def load_data_llm(_self):

        """
        Loads and preprocesses data including RN_df, X_sell_df, and claims_df from session state for Sales Pitches.

        Returns:
        --------
        tuple
            Tuple containing preprocessed DataFrames: RN_df, X_sell_df, claims_df, claims_df_PT, and top3_frequent_combined.
        """

        RN_df = st.session_state.RN_df
        X_sell_df = st.session_state.cross_sell_sn_df
        claims_df = st.session_state.raw_claims_df

        RN_df = RN_df[['OLD_POLICY_NO', 'Conversion Status', 'Booked Premium', 'Quoted Premium',  'Zone', 'State','Expiring Premium', 'Occupancy', 'Risk category', 
                    'CONVERSIONTYPE', 'SI', 'SI Band', 'Inception Month', 'Binding Month', 'Spillover', 'Final Channel', 'Main Channel', 'Producer Code']]
        RN_df.rename(columns={'OLD_POLICY_NO': 'POLICY_NUMBER'}, inplace= True)
        RN_df.rename(columns={'Producer Code': 'PRODUCER_CD'}, inplace= True)
        RN_df.rename(columns={'Zone': 'New_Zone'}, inplace= True)
        RN_df.dropna(subset=['POLICY_NUMBER'], inplace=True)
        RN_df['POLICY_NUMBER'] = RN_df['POLICY_NUMBER'].astype(int)

        RN_df = RN_df[(RN_df['New_Zone'].isin(['Mumbai', 'Pune'])) &
                    (RN_df['State'] == 'Maharashtra')]

        RN_df['Inception Month'] = pd.to_datetime(RN_df['Inception Month'], format='%b-%y', errors='coerce')
        RN_df.loc[RN_df['Inception Month'].notnull(), 'Expiry Date'] = RN_df['Inception Month'] + pd.DateOffset(days=365)
        RN_df.dropna(subset=['PRODUCER_CD'], inplace= True)

        RN_df = RN_df.reset_index(drop=True)

        X_sell_df = X_sell_df[['POLICY_NO', 'base_lob', 'CLIENTNAME', 'TIMES_RENEWED_CNT', 'PRODUCER_CD', 'NEW_LOB', 'CL_ZONE', 'FINAL_CHANNEL', 'MAIN_CHANNEL', 'SUM_INSURED', 'AIGC', 'AIGC-EC', 'CASUALTY', 'ENGG.', 'FL', 'GHI', 'GPA', 'MARINE-NON SMP', 'MARINE-SMP', 'AIGC_Prem', 'AIGC-EC_Prem',
                                'CASUALTY_Prem', 'ENGG._Prem', 'FL_Prem', 'GHI_Prem', 'GPA_Prem', 'MARINE-NON SMP_Prem', 'MARINE-SMP_Prem', 'occupancy_name',
                                'PREMIUMAMOUNT']]

        X_sell_df[X_sell_df['base_lob'] == 'AIGC'].drop(columns=['base_lob'])
        columns_to_fill = ['AIGC-EC', 'CASUALTY', 'ENGG.', 'FL', 'GHI', 'GPA', 'MARINE-NON SMP']

        X_sell_df[columns_to_fill] = X_sell_df[columns_to_fill].fillna(0).astype(int)
        # Create 'cross_sold' column
        X_sell_df['cross_sold'] = X_sell_df[columns_to_fill].max(axis=1).apply(lambda x: 1 if x > 0 else 0)

        # Filter rows where 'cross_sold' is 1 and drop the column
        undr_smpld_df = X_sell_df[X_sell_df['cross_sold'] == 1].drop(columns=['cross_sold']).reset_index(drop=True)

        # Group by 'occupancy_name' and sum the occurrences
        grouped = undr_smpld_df.groupby('occupancy_name')[columns_to_fill].sum()

        # Get the top 3 most frequent columns and their counts for each 'occupancy_name'
        top3_frequent_combined = grouped.apply(
            lambda row: pd.Series({
                'most_frequent_columns': row.nlargest(3).index.tolist(),
                'counts': row.nlargest(3).values.tolist()
            }), axis=1
        ).reset_index()

        X_sell_df = X_sell_df[(X_sell_df['CL_ZONE'].isin(['PUNE', 'MUMBAI']))]
        X_sell_df.rename(columns={'POLICY_NO': 'POLICY_NUMBER'}, inplace= True)

        # Convert the relevant columns to integers
        columns_to_convert = ['AIGC', 'AIGC-EC', 'CASUALTY', 'ENGG.', 'FL', 'GHI', 'GPA', 'MARINE-NON SMP', 'MARINE-SMP']
        X_sell_df[columns_to_convert] = X_sell_df[columns_to_convert].replace([np.inf, -np.inf, np.nan], 0)
        X_sell_df[columns_to_convert] = X_sell_df[columns_to_convert].astype(int)

        # Now you can use your original function without modification
        def get_xsells(row):
            xsells = []
            for col in columns_to_convert:
                if row[col] == 1:
                    xsells.append(col)

            if xsells:
                return tuple(xsells)
            else:
                return "No Cross selling"

        # Apply the function to the DataFrame
        X_sell_df['Cross_Sold'] = X_sell_df.apply(get_xsells, axis=1)

        def extract_prem_values(row):
            prem_dict = {}
            for col in ['AIGC_Prem', 'AIGC-EC_Prem', 'CASUALTY_Prem', 'ENGG._Prem', 'FL_Prem', 'GHI_Prem', 'GPA_Prem', 'MARINE-NON SMP_Prem', 'MARINE-SMP_Prem']:
                if isinstance(row[col], float) and not pd.isna(row[col]):
                    prem_dict[col] = row[col]
            if prem_dict:  # Check if dictionary is not empty
                return prem_dict
            else:
                return "No Cross Sell"

        # Apply the function across each row of the DataFrame and store the result in 'XSell_Prem' column
        X_sell_df['Cross_sold_prem'] = X_sell_df.apply(extract_prem_values, axis=1)
        X_sell_df = X_sell_df.reset_index(drop=True)

        claims_df = claims_df[['Claim_Number', 'Policy_No', 'PRODUCER_CD', 'Insured_Name', 'Claimant_Name', 'Product_Name_As_Per_Register', 'Loss_Date', 'Pol_Incept_Date', 'Month',
                            'Cause_Of_Loss', ' NLI ',  'GLI', 'Branch Type_Comm',  'City_Name', 'New_Zone', 'State', 'Producer_Name', 'Revised_SME']]

        claims_df = claims_df[(claims_df['Branch Type_Comm'] == 'top 15') & 
                (claims_df['State'] == 'maharashtra') & 
                (claims_df['City_Name'].isin(['mumbai', 'pune'])) &
                (claims_df['Revised_SME'] == 'SME')]
        
        # Convert the columns to integers
        claims_df['Policy_No'] = claims_df['Policy_No'].astype(int)
        claims_df['PRODUCER_CD'] = claims_df['PRODUCER_CD'].astype(int)
        claims_df.rename(columns={'Policy_No': 'POLICY_NUMBER'}, inplace= True)
        claims_df.dropna(subset=['Pol_Incept_Date'], inplace = True)
    
        claims_df['GLI'] = claims_df['GLI'].astype(float).astype(int)
        claims_df = claims_df.reset_index(drop=True)

        claims_df_PT = claims_df.pivot_table(index= ['POLICY_NUMBER', 'Claim_Number'], values= ['GLI'], aggfunc= 'sum').reset_index()

        return RN_df, X_sell_df, claims_df, claims_df_PT, top3_frequent_combined 

    def sales_pitch(self, pol_num : int):

        """
        Generates a sales pitch for a specific policy number by extracting and analyzing customer information,
        renewal details, cross-selling opportunities, and claims history.

        Parameters:
        -----------
        pol_num : int
            Policy number for which the sales pitch is generated.

        Returns:
        --------
        dict
            Results of the sales pitch including recommendations for renewal, upselling, and cross-selling.

        """

        RN_df, X_sell_df, claims_df, claims_df_PT, top3_frequent_combined = self.load_data_llm()
        new_db = st.session_state.new_db

        class Sales_Pitch(BaseModel):
            Renewal_Pitch: str = Field(description="recommendation and sales pitch for the renewal for the policy")
            Upselling_Pitch: str = Field(description="recommendation and sales pitch for the upselling of other coverages for the policy")
            Cross_Sell_Pitch: str = Field(description="recommendation and sales pitch for the Cross Selling other Line of Businesses for the policy")

        # And a query intented to prompt a language model to populate the data structure.
        customer_info_RN = f"""
                Customer Renewal Information:
                The booked, quoted and Expiring premium of the customer is Rs. {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Booked Premium'].iloc[0])},  Rs. {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Quoted Premium'].iloc[0])}, 
                and Rs. {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Expiring Premium'].iloc[0])} respectively,
                Zone: {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['New_Zone'].iloc[0])} and State: {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['State'].iloc[0])},
                Occupancy covered in the policy is {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['occupancy_name'].iloc[0]}. The Risk category out of (Low, High-Risk, Critical) is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Risk category'].iloc[0])},
                CONVERSION TYPE is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['CONVERSIONTYPE'].iloc[0])}, 
                Sum Insured is Rs. {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['SI'].iloc[0])}, Sum Insured band out of ('50 Lakhs to 2.5 Cr', '10 to 25 Lakhs', '2.5 to 5 Cr', '25 to 50 Lakhs', '10 to 25 Cr', '25 to 50 Cr', '5 to 10 Cr', '>50Cr', '<=10 Lakhs') is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['SI Band'].iloc[0])},
                Policy Expiry date is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Expiry Date'].iloc[0])}, 
                Final Channel and Main Channel are {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Final Channel'].iloc[0])} and {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Main Channel'].iloc[0])} respectively.

        """

        customer_info_Xsell = f"""
                Cross Selling Information:
                Base Line of Business(LOB) of the customer is {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['NEW_LOB'].iloc[0]}.
                Age of the policy is {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['TIMES_RENEWED_CNT'].iloc[0]} years. (must include in summary)
                Premium Amount is Rs. {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['PREMIUMAMOUNT'].iloc[0]* 10**7} 
                These coverages has been cross sold before : ({X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['Cross_Sold'].iloc[0]}), "No cross selling" indicates furthur possibiliy of selling.
                and the cross sold premium is {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['Cross_sold_prem'].iloc[0]} (in crores)
        """

        customer_info_Claims = f"""
        Customer Claims Information:
        The customer has purchased the {str(claims_df[claims_df['POLICY_NUMBER'] == pol_num]['Product_Name_As_Per_Register'].unique())} product from TATA AIG.
        cause of loss for the policy claim is {str(claims_df[claims_df['POLICY_NUMBER'] == pol_num]['Cause_Of_Loss'].unique())}, 
        the total number of previous claims of the customer is {claims_df_PT[claims_df_PT['POLICY_NUMBER'] == int(pol_num)].Claim_Number.nunique()} (if the number of claims > 0, mention the exact number of claims then create an urge or fear of renewal of the policy), 
        and Gross Loss Incurred (GLI) is Rs. {claims_df_PT[claims_df_PT['POLICY_NUMBER'] == int(pol_num)].GLI.sum()}
        """

        str_occu = str(X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['occupancy_name'].iloc[0])

        cross_selling_LOBs = f"""
        {top3_frequent_combined[top3_frequent_combined['occupancy_name'] == str_occu]['most_frequent_columns'].iloc[0]}
        """

        search_cover = f"""
            Occupancy covered in the policy is {str(X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['occupancy_name'].iloc[0])}, 

        """

        print(search_cover)

        try: 
            retriever = new_db.as_retriever(search_type="similarity_score_threshold",
                                            search_kwargs={'score_threshold': 0.41, 'k': 6})

            docs = retriever.invoke(search_cover)
            
        except:
            docs = "No products found for this occupancy"
            
        covers_VS = {}
            
        # Iterate through each document and construct the desired structure
        for idx, doc in enumerate(docs, start=1):
            product_key = f"coverage_{idx}"
            print(type(doc))
            # cover = doc.metadata['Cover_group']
            cover = doc.metadata.get('Cover_group', "Unknown")

            covers_VS[product_key] = cover

        template_customer_info = """ 
        You are an expert assistant in the TATA AIG General Insurance Industry.
        Given the following customer information for a TATA AIG General Insurance policy, recommend impactful insurance tips for the policy renewal for the TATA AIG Customer to the TATA AIG's Agent.
        Create a sales pitch for the TATA AIG's Agent, so that he/she can pitch it to the customer/client.
        Ensure the recommendations create an urge for the customer to renew the policy.
        Check if the Sum Insured is adequate based on the customer's occupancy and provide suggestions accordingly.
        Restrict recommendations to TATA AIG General Insurance policies only, excluding other insurance companies.
        Include exact numbers in Indian Rupees in the recommendations to clearly explain to the agent. Give the comparison of the previous policy and what will happen after policy renewal.

        **Format Instructions:**
        \n{format_instructions}\n
        - Always present the final response in **bullet points**.
        - Highlight **numerical values** in bold.
        - Use *italics* for emphasis and clarity.

        Customer previous claims information: {customer_info_Claims} 
        - Based on this Claims Information, provide suggestions for renewal as well as cross-selling for the policy.

        Customer Renewal Information: {customer_info_RN} 
        - Include intuitive suggestions for the sales agent and give risk management tips based on the customer information.

        Upselling Information: {covers_VS} 
        - Customer doesn't have these coverages; always suggest these covers to the customer only for upselling.
        - Include these coverages in the renewal pitch.

        Customer Cross Selling Information: {customer_info_Xsell} 
        - Consider these LOBs only for cross-selling: {cross_selling_LOBs}.
        - Strictly include these LOBs only for cross-selling, do not suggest any external LOBs, and explain why these LOBs are necessary to cross-sell to the customer.

        Response:
        - Bullet point 1\n
        - Bullet point 2\n
        - Bullet point 3\n
        ...

        """

        parser = JsonOutputParser(pydantic_object=Sales_Pitch)

        prompt = PromptTemplate(
            template=template_customer_info,
            input_variables=["customer_info_RN",  "customer_info_Claims", "covers_VS", "customer_info_Xsell", "cross_selling_LOBs"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.llm | parser

        print(covers_VS)

        print(cross_selling_LOBs)


        results_ = chain.invoke({"customer_info_RN": customer_info_RN,
                            "customer_info_Claims": customer_info_Claims,
                            "covers_VS": covers_VS, 
                            "customer_info_Xsell": customer_info_Xsell, 
                            "cross_selling_LOBs": cross_selling_LOBs})
        return results_
    
    # @st.cache_resource
    def chat_with_data(self, pol_num : int):

        """
        Generates a prompt for chat with data interaction based on customer information,
        renewal details, cross-selling opportunities, and claims history.

        Parameters:
        -----------
        pol_num : int
            Policy number for which the chat interaction is generated.

        Returns:
        --------
        retriever_rag
            A retriever object for performing RAG (retrieval-augmented generation) based on the constructed prompt.
        """
        
        RN_df = st.session_state.RN_df
        X_sell_df = st.session_state.cross_sell_sn_df
        claims_df = st.session_state.raw_claims_df
        RN_df, X_sell_df, claims_df, claims_df_PT, top3_frequent_combined = self.load_data_llm()
        new_db = st.session_state.new_db

        # And a query intented to prompt a language model to populate the data structure.
        customer_info_RN = f"""
                Customer Renewal Information:
                The booked, quoted and Expiring premium of the customer is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Booked Premium'].iloc[0])},  {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Quoted Premium'].iloc[0])}, 
                and {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Expiring Premium'].iloc[0])} respectively,
                Zone: {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['New_Zone'].iloc[0])} and State: {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['State'].iloc[0])},
                Occupancy covered in the policy is {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['occupancy_name'].iloc[0]}. The Risk category out of (Low, High-Risk, Critical) is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Risk category'].iloc[0])},
                CONVERSION TYPE is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['CONVERSIONTYPE'].iloc[0])}, 
                Sum Insured is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['SI'].iloc[0])}, Sum Insured band out of ('50 Lakhs to 2.5 Cr', '10 to 25 Lakhs', '2.5 to 5 Cr', '25 to 50 Lakhs', '10 to 25 Cr', '25 to 50 Cr', '5 to 10 Cr', '>50Cr', '<=10 Lakhs') is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['SI Band'].iloc[0])},
                Policy Expiry date is {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Expiry Date'].iloc[0])}, 
                Final Channel and Main Channel are {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Final Channel'].iloc[0])} and {str(RN_df[RN_df['POLICY_NUMBER'] == pol_num]['Main Channel'].iloc[0])} respectively.

        """

        customer_info_Xsell = f"""
                Cross Selling Information:
                Base Line of Business(LOB) of the customer is {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['NEW_LOB'].iloc[0]}.
                Age of the policy is {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['TIMES_RENEWED_CNT'].iloc[0]} years. (must include in summary)
                Premium Amount is {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['PREMIUMAMOUNT'].iloc[0]} (in crores)
                These coverages has been cross sold before : ({X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['Cross_Sold'].iloc[0]}), "No cross selling" indicates furthur possibiliy of selling.
                and the cross sold premium is {X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['Cross_sold_prem'].iloc[0]} (in crores)
                
        """

        customer_info_Claims = f"""
        Customer Claims Information:
        The customer has purchased the {str(claims_df[claims_df['POLICY_NUMBER'] == pol_num]['Product_Name_As_Per_Register'].unique())} product from TATA AIG.
        cause of loss for the policy claim is {str(claims_df[claims_df['POLICY_NUMBER'] == pol_num]['Cause_Of_Loss'].unique())}, 
        the total number of previous claims of the customer is {claims_df_PT[claims_df_PT['POLICY_NUMBER'] == int(pol_num)].Claim_Number.nunique()} (if the number of claims > 0, mention the exact number of claims then create an urge or fear of renewal of the policy), 
        and Gross Loss Incurred (GLI) is {claims_df_PT[claims_df_PT['POLICY_NUMBER'] == int(pol_num)].GLI.sum()}
        """

        str_occu = str(X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['occupancy_name'].iloc[0])

        cross_selling_LOBs = f"""
        {top3_frequent_combined[top3_frequent_combined['occupancy_name'] == str_occu]['most_frequent_columns'].iloc[0]}
        """

        search_cover = f"""
            Occupancy covered in the policy is {str(X_sell_df[X_sell_df['POLICY_NUMBER'] == int(pol_num)]['occupancy_name'].iloc[0])}, 

        """

        print(search_cover)

        try: 
            retriever_cover = new_db.as_retriever(search_type="similarity_score_threshold",
                                            search_kwargs={'score_threshold': 0.41, 'k': 5})

            docs = retriever_cover.invoke(search_cover)
            
        except:
            docs = "No products found for this occupancy"
            
        covers_VS = {}
            
        # Iterate through each document and construct the desired structure
        for idx, doc in enumerate(docs, start=1):
            product_key = f"coverage_{idx}"
            cover = doc.metadata['Cover_group']
            covers_VS[product_key] = cover

        template_customer_info = """ 

                    The customer previous claims information is {customer_info_Claims}.\n
                    Customer Renewal Inforamtion: {customer_info_RN}.\n
                    Upselling Inforamtion: {covers_VS} (Customer doesn't have these coverages, suggest these covers to the customer only for upselling, include these coverages in the renewal pitch).\n
                    Customer Cross Selling Information: {customer_info_Xsell} and consider these LOBs only for cross selling{cross_selling_LOBs}.\n

                """
        promptt = template_customer_info.format(customer_info_Claims= customer_info_Claims,
                                            customer_info_RN = customer_info_RN,
                                            covers_VS = covers_VS,
                                            customer_info_Xsell = customer_info_Xsell,
                                            cross_selling_LOBs = cross_selling_LOBs)

        docs_ =  [Document(page_content=promptt)]

        # Save to disk
        vectorstore = FAISS.from_documents(
                            documents = docs_,                 # Data
                            embedding = self.embeddings    # Embedding model
                            )
        vectorstore.save_local("./faiss_index_chat_openai")

        # Load from disk
        vectorstore_disk = FAISS.load_local("./faiss_index_chat_openai", self.embeddings, allow_dangerous_deserialization=True)
        retriever_rag = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

        return retriever_rag
