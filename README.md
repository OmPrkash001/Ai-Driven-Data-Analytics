# AI-Integrated Data Analyst Project

## Project Description
This project is a complete, production-ready end-to-end AI-Integrated Data Analyst system. It features automated data ingestion, cleaning, feature engineering, and machine learning model training with automatic detection of regression or classification tasks. The system includes a Streamlit dashboard for interactive data analysis, visualization, and AI-powered insights using Google's Gemini models.

## Features
- **Automated ETL**: Ingestion, cleaning, missing value handling, and outlier capping.
- **Feature Engineering**: Date-time decomposition and automated feature preparation.
- **Auto-ML**: Automatically detects regression or classification problems and trains XGBoost or LightGBM models respectively.
- **Interactive Dashboard**: Streamlit app for data upload, EDA, and model training.
- **AI Integration**: "Ask your dataset" feature using Google Gemini to provide natural language insights.
- **Dockerized**: Ready for deployment with Docker.
- **CI/CD**: GitHub Actions workflow for continuous integration.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-data-analyst-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   Create a `.env` file in the root directory and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Running the Dashboard
To start the Streamlit application:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

### Running ETL Standalone
```bash
python src/etl.py path/to/your/data.csv
```

### Training Model Standalone
You can import the `run_training` function from `src.train_model` in a script or notebook.

## Docker Usage

1. **Build the image:**
   ```bash
   docker build -t ai-data-analyst .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 --env-file .env ai-data-analyst
   ```

## Folder Structure
```
ai-data-analyst-project/
├── data/               # Data storage
│   └── raw/            # Raw uploaded files
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
│   ├── etl.py          # ETL pipeline
│   ├── features.py     # Feature engineering
│   ├── train_model.py  # Model training
│   ├── predict.py      # Prediction logic
│   └── utils.py        # Utilities
├── app.py              # Streamlit application
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies
└── .github/            # CI/CD configuration
```
