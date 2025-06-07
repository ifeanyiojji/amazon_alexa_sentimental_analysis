# Alexa Reviews Sentiment Analysis

This project analyzes customer reviews for Amazon Alexa devices to predict sentiment (positive/negative) using machine learning models.

---

## Project Overview

- **Goal:** Build a sentiment analysis model to classify Alexa user reviews as positive or negative.
- **Dataset:** Amazon Alexa reviews provided in `.tsv` and `.csv` formats.
- **Model:** XGBoost classifier trained on vectorized text data.
- **Tools:** Python, Scikit-learn, XGBoost, Streamlit for deployment.

---

## Files and Structure

- `app.py` — Main application script for running the sentiment prediction app (likely with Streamlit).
- `amazon_alexa.tsv` — Primary dataset containing Alexa user reviews and sentiments.
- `alexa.csv` — Additional or processed data related to Alexa reviews.
- `countVectorizer.pkl` — Serialized CountVectorizer for transforming review text into features.
- `scaler.pkl` — Serialized scaler object used for feature scaling.
- `xgb.pkl` — Trained XGBoost model for sentiment prediction.
- `sentimental analysis.ipynb` — Jupyter notebook used for data exploration, preprocessing, and model training.
- `requirements.txt` — Lists all Python dependencies needed to run the project.
- `README.md` — This documentation file.

---

## How to Run

1. **Clone the repository:**

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run The App:**
   ```bash
   streamlit run app.py
   ```


## Usage

- Input any Amazon Alexa review in the app.
- The app will preprocess the input, vectorize it using the saved `countVectorizer.pkl`, scale features using `scaler.pkl`, and predict the sentiment with the XGBoost model.
- The predicted sentiment (positive or negative) will be displayed.

---

## Limitations & Known Issues

- Model performance depends on training data quality.
- Only works with reviews similar in style to the training set.
- No support yet for multi-language reviews.
- The app does not update model or vectorizer dynamically.

---

## Future Improvements

- Add support for real-time model retraining.
- Expand dataset to include more diverse reviews.
- Add multi-language sentiment analysis support.
- Improve UI/UX for better user experience.

---

## Author

Developed by Ifeanyi Ojji  
Connect on [LinkedIn](https://linkedin.com/in/ifeanyiojji) | [GitHub](https://github.com/ifeanyiojji)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Scikit-learn](https://scikit-learn.org/) for machine learning tools  
- [XGBoost](https://xgboost.readthedocs.io/) for the gradient boosting model  
- [Streamlit](https://streamlit.io/) for the interactive app  
- Dataset sourced from Amazon Alexa reviews
