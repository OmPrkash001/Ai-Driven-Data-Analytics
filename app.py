import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
import google.generativeai as genai
from src.etl import run_etl
from src.train_model import run_training
from src.predict import make_predictions

# Load environment variables
load_dotenv()

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("🤖 AI-Integrated Data Analyst")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    data_path = os.path.join("data", "raw", uploaded_file.name)
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"File uploaded: {uploaded_file.name}")
    
    # ETL
    st.header("1. Data Overview & ETL")
    if st.button("Run ETL Pipeline"):
        with st.spinner("Running ETL..."):
            df = run_etl(data_path)
            st.session_state['df'] = df
            st.success("ETL Completed!")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Summary")
            st.write(df.describe())
        with col2:
            st.subheader("Missing Values")
            st.write(df.isnull().sum())
            
        # EDA
        st.header("2. Exploratory Data Analysis")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        selected_col = st.selectbox("Select column for distribution", numeric_cols)
        
        if selected_col:
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig)
            
        # Model Training
        st.header("3. Model Training")
        target_col = st.selectbox("Select Target Column", df.columns)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                metrics = run_training(df, target_col)
                st.session_state['model_metrics'] = metrics
                st.success("Training Completed!")
                
        if 'model_metrics' in st.session_state:
            st.json(st.session_state['model_metrics'])
            
        # AI Insights
        st.header("4. AI Insights & Chat")
        if api_key:
            user_query = st.text_input("Ask a question about your dataset:")
            if user_query:
                # Prepare context
                context = f"Dataset columns: {list(df.columns)}\nSummary stats: {df.describe().to_string()}"
                if 'model_metrics' in st.session_state:
                    context += f"\nModel Metrics: {st.session_state['model_metrics']}"
                
                prompt = f"""
                You are an expert data analyst. 
                Context:
                {context}
                
                User Question: {user_query}
                
                Provide a clear, concise answer or insight.
                """
                
                with st.spinner("Generating insight..."):
                    try:
                        model = genai.GenerativeModel('gemini-flash-latest')
                        response = model.generate_content(prompt)
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error generating insight: {e}")
        else:
            st.warning("Please set GEMINI_API_KEY in .env file to use AI features.")

else:
    st.info("Please upload a CSV file to begin.")
