import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Alexa Sentiment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5;
    }
    .st-bw {
        background-color: #FFFFFF;
    }
    .css-18e3th9 {
        padding: 2rem 5rem;
    }
    .header-text {
        font-size: 2.5rem !important;
        color: #2F4F4F;
        text-align: center;
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px;
    }
    .warning {
        color: #FF4B4B;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    xgb_model = pickle.load(open('xgb.pkl', 'rb'))
    cv = pickle.load(open('countVectorizer.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return xgb_model, cv, scaler

xgb_model, cv, scaler = load_models()

# Main app
st.markdown('<p class="header-text">Amazon Alexa Reviews Sentiment Analysis</p>', unsafe_allow_html=True)
st.markdown("---")

st.markdown("### 📥 Download Sample Dataset")
st.markdown(
    "[Click here to download the dataset](https://drive.google.com/file/d/1ZWwT73egLbFXqpnOsOdoMyOmXYz64kyX/view?usp=drive_link)",
    unsafe_allow_html=True,
)
st.markdown("---")


# File upload section
uploaded_file = st.file_uploader("Upload your Amazon reviews CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'verified_reviews' not in df.columns:
            st.error("CSV file must contain 'verified_reviews' column")
        else:
            # Clean data
            initial_count = len(df)
            df = df.dropna(subset=['verified_reviews'])  # Remove rows with missing reviews
            df['verified_reviews'] = df['verified_reviews'].astype(str)  # Ensure all values are strings
            cleaned_count = len(df)
            
            if cleaned_count == 0:
                st.error("No valid reviews found after cleaning!")
            else:
                # Show cleaning warning if rows were removed
                if cleaned_count != initial_count:
                    st.warning(f"Removed {initial_count - cleaned_count} rows with missing reviews")
                
                # Preprocess text
                X = cv.transform(df['verified_reviews'])
                X_scaled = scaler.transform(X.toarray())
                
                # Predictions
                predictions = xgb_model.predict_proba(X_scaled)[:, 1]
                df['sentiment'] = ['Positive' if prob >= 0.5 else 'Negative' for prob in predictions]
                df['sentiment_score'] = predictions
                
                # Display results
                st.markdown("### Analysis Results")
                
                # Metrics in columns
                col1, col2, col3 = st.columns(3)
                card_style = """
                <div style="
                background: #333; 
                border-radius: 10px; 
                padding: 20px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                margin: 10px;
                color: white;
                text-align: center;
                ">
                <h3 style="margin-bottom:5px;">{title}</h3>
                <h2>{value}</h2>
                </div>
                """
                with col1:
                    st.markdown(card_style.format(title="📈 Average Sentiment", value=f"{np.mean(predictions):.2%}"), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(card_style.format(title="📊 Positive Reviews", value=f"{(predictions >= 0.5).sum()}"), unsafe_allow_html=True)

                
                with col3:
                    st.markdown(card_style.format(title="📦 Valid Reviews", value=f"{cleaned_count}"), unsafe_allow_html=True)
                
                # Visualizations
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Sentiment Distribution")
                    sentiment_dist = df['sentiment'].value_counts()
                    st.bar_chart(sentiment_dist)
                
                with col2:
                    st.markdown("### Sentiment Score Distribution")
                    st.line_chart(df['sentiment_score'])
                
                # Raw data preview
                st.markdown("---")
                st.markdown("### Analyzed Data Preview")
                st.dataframe(df[['verified_reviews', 'sentiment', 'sentiment_score']].head(10))
                
                # Sample positive/negative reviews
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 😊 Top Positive Review")
                    positive_sample = df[df['sentiment_score'] == df['sentiment_score'].max()].iloc[0]
                    st.markdown(f'"{positive_sample["verified_reviews"]}"')
                    st.markdown(f'**Score:** {positive_sample["sentiment_score"]:.2%}')
                
                with col2:
                    st.markdown("### 😞 Top Negative Review")
                    negative_sample = df[df['sentiment_score'] == df['sentiment_score'].min()].iloc[0]
                    st.markdown(f'"{negative_sample["verified_reviews"]}"')
                    st.markdown(f'**Score:** {negative_sample["sentiment_score"]:.2%}')

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px;">
        <h3>📤 Upload a CSV file to analyze Amazon Alexa reviews</h3>
        <p>Your CSV file should contain a 'verified_reviews' column with customer feedback</p>
    </div>
    """, unsafe_allow_html=True)