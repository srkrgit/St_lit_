import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from data_preprocess import DataPreprocess

image_path = "https://upload.wikimedia.org/wikipedia/commons/0/03/TATA_AIG_logo.png"


class MainApp:

    def __init__(self, embeddings, llm):

        """
        Initialize the MainApp class with embeddings and a language model (llm).

        Args:
        - embeddings (object): Embeddings object for data preprocessing.
        - llm (object): Language model object for generating responses.

        This method initializes necessary attributes and sets up session state variables.
        """

        self.embeddings = embeddings
        self.llm = llm

        self.data_preprocess = DataPreprocess(self.embeddings , self.llm)


    def run(self):

        """
        Method to run the Streamlit application.

        This method sets up the Streamlit interface, including headers, input fields, buttons, and output displays.
        """

            
        header_html = f"""
            <div style="display: flex; justify-content: center; align-items: center;">
                <img src="{image_path}" style="height: 100px; margin-right: 10px;">
                <h2>Sales Pitch Assistant</h2>
            </div>
        """

        header = st.container()
        # Display the custom header using markdown
        header.markdown(header_html, unsafe_allow_html=True)
        header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

        ### Custom CSS for the sticky header
        st.markdown(
            """
        <style>
            div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                position: sticky;
                top: 3.5rem;
                background-color: white;
                z-index: 999;
            }
            .fixed-header {

            }
        </style>
            """,

            unsafe_allow_html=True
        )

        Prod_CD = st.text_input("Enter Prod Cd")
        pol_num = st.text_input("Enter Pol no")

        col1, col2, col3, col4 = st.columns(4)

        # Initialize session state variables
        if 'show_policy' not in st.session_state:
            st.session_state['show_policy'] = False

        if 'show_claims' not in st.session_state:
            st.session_state['show_claims'] = False

        if 'show_pitch' not in st.session_state:
            st.session_state['show_pitch'] = False

        if 'show_chat' not in st.session_state:
            st.session_state['show_chat'] = False

        if 'res' not in st.session_state:
            st.session_state['res'] = None

        # Fetch data if text inputs change
        if st.session_state.get('prev_Prod_CD') != Prod_CD or st.session_state.get('prev_pol_num') != pol_num:
            st.session_state['show_policy'] = False
            st.session_state['show_claims'] = False
            st.session_state['show_pitch'] = False
            st.session_state['show_chat'] = False
            st.session_state['res'] = None
            st.session_state['prev_Prod_CD'] = Prod_CD
            st.session_state['prev_pol_num'] = pol_num

        with col1:
            if st.button("Policy Details", use_container_width=True, type='primary'):
                st.session_state['show_policy'] = True
                st.session_state['show_claims'] = False  # Reset claims details display
                st.session_state['show_pitch'] = False
                st.session_state['show_chat'] = False

        with col2:
            if st.button("Claims Details", use_container_width=True, type='primary'):
                st.session_state['show_claims'] = True
                st.session_state['show_policy'] = False  # Reset policy details display
                st.session_state['show_pitch'] = False
                st.session_state['show_chat'] = False

        with col3:
            if st.button("Sales Pitches", use_container_width=True, type='primary'):
                st.session_state['show_pitch'] = True
                st.session_state['show_policy'] = False
                st.session_state['show_claims'] = False 
                st.session_state['show_chat'] = False
                # Fetch new response only if not already fetched
                if st.session_state['res'] is None and pol_num:
                    st.session_state['res'] = self.data_preprocess.sales_pitch(int(pol_num))

        with col4:
            if st.button("Chat", use_container_width=True, type='primary'):
                st.session_state['show_chat'] = True
                st.session_state['show_claims'] = False
                st.session_state['show_policy'] = False  # Reset policy details display
                st.session_state['show_pitch'] = False

        # Display the policy details table if the button was clicked
        if st.session_state['show_policy']:
            try:
                if Prod_CD and pol_num:
                    st.table(self.data_preprocess.prod_cust(Prod_CD, pol_num))
            except:
                st.warning("This producer does not contain this policy number!")

        # Display the claims details dataframe if the button was clicked
        if st.session_state['show_claims']:
            try:
                if pol_num:
                    st.table(self.data_preprocess.claims_info(int(pol_num)))
            except:
                st.warning("This producer does not contain this policy number!")
                    
        if st.session_state['show_pitch']:
            response = st.session_state['res']

            if response:
                # Define the style for the columns
                st.markdown("""
                                <style>
                                .column {
                                    padding: 20px;
                                    margin: 10px;
                                    border: 1px solid black;
                                    border-radius: 10px;
                                    background-color: #F5EFE6;
                                    box-sizing: border-box;
                                }
                                .title {
                                    font-weight: bold;
                                    text-align: center;
                                    margin-bottom: 10px;
                                }
                                .break-line {
                                    border-top: 1px solid black;
                                    margin-top: 2px;
                                    margin-bottom: 10px;
                                }
                                </style>
                                """, unsafe_allow_html=True)

                renewal_pitch = response['Renewal_Pitch']
                crossSell_pitch = response['Cross_Sell_Pitch']

                # Create two columns in the second row
                col2, col3 = st.columns(2)

                # Add text and title to the second row columns
                with col2:
                    # st.markdown("**Renewal Pitch**")
                    st.markdown("<h1 style='font-size:26px; font-weight: bold; text-align: center; text-decoration: underline;'>Renewal Pitch</h1>", unsafe_allow_html=True)
                    st.markdown(renewal_pitch)

                with col3:
                    # st.markdown("**Cross Sell Pitch**")
                    st.markdown("<h1 style='font-size:26px; font-weight: bold; text-align: center; text-decoration: underline;'>Cross Sell Pitch</h1>", unsafe_allow_html=True)
                    st.markdown(crossSell_pitch)

        if st.session_state['show_chat']:

            if pol_num:
                # Prompt template to query Gemini
                llm_prompt_template = """You are an expert General Insurance assistant for question-answering.
                You provide helpful insights for the agents from the given customer data which contains insurance policy details, claims details, cross sellng details, etc.
                Use the following context to answer the question. You can use external knowledge to answer any factual questions.
                If you don't know the answer, just say that you don't know.
                Keep the answer relevant to the given question, do not give extra information.\n
                Suggest a few(one or two) followup questions to be asked at the end of the response in bullet points.\n
                Question: {question} \nContext: {context} \nAnswer:"""

                llm_prompt = PromptTemplate.from_template(llm_prompt_template)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                retriever_rag = self.data_preprocess.chat_with_data(int(pol_num))

                rag_chain = (
                    {"context": retriever_rag | format_docs, "question": RunnablePassthrough()}
                    | llm_prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": "Sales Assistant Chatbot"
                        }
                    ]

                # Display chat messages from history on app rerun
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Process and store Query and Response
                def llm_function(query):
                    response = rag_chain.invoke(query)

                    # Displaying the Assistant Message
                    with st.chat_message("assistant"):
                        st.markdown(response)

                    # Storing the User Message
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": query
                        }
                    )

                    # Storing the Assistant Message
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response
                        }
                    )

                # Accept user input
                query = st.chat_input("Enter you query here...")

                # Calling the Function when Input is Provided
                if query:
                    # Displaying the User Message
                    with st.chat_message("user"):
                        st.markdown(query)

                    llm_function(query)
