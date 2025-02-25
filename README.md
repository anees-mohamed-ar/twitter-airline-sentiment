# Twitter Airline Sentiment Analysis

![Sentiment Analysis Banner](https://img.shields.io/badge/NLP-Sentiment%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Project Overview

This project analyzes Twitter data related to major US airlines to automatically classify customer sentiment as positive, negative, or neutral. Using machine learning and natural language processing techniques, we build models that can understand and categorize customer feedback, enabling airlines to identify areas for improvement and respond to customer concerns more effectively.

## 🎯 Objectives

- Build a sentiment analysis model for airline-related tweets
- Compare multiple machine learning algorithms for text classification
- Create visualizations to understand sentiment patterns across different airlines
- Develop a reusable pipeline for predicting sentiment of new tweets

## 📊 Dataset

The project uses the Twitter US Airline Sentiment dataset available on Kaggle:

- **Source**: [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Size**: Approximately 14,000 tweets
- **Time Period**: February 2015
- **Content**: Tweets about six major US airlines (American, Delta, Southwest, United, US Airways, Virgin America)
- **Labels**: Positive, Negative, and Neutral sentiment
- **Features**: Tweet text, airline, sentiment, confidence scores, and reasons for negative sentiment

## 🔧 Technologies Used

- **Python 3.7+**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **NLTK**: Natural language processing tasks
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Data visualization
- **Regular Expressions (re)**: Text cleaning

## 🚀 Installation & Setup

1. Clone this repository:
   ```
   git clone https://github.com/anees-mohamed-ar/twitter-airline-sentiment.git
   cd twitter-airline-sentiment
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) and place the `Tweets.csv` file in the project directory.

## 📝 Project Structure

```
twitter-airline-sentiment/
│
├── data/
│   └── Tweets.csv           # The dataset file
│
├── notebooks/
│   └── exploration.ipynb    # Jupyter notebook for data exploration
│
├── src/
│   ├── preprocessing.py     # Text preprocessing functions
│   ├── feature_extraction.py # Feature extraction code
│   ├── model_training.py    # Model training and evaluation
│   └── predict.py           # Prediction function for new tweets
│
├── models/
│   ├── best_sentiment_model.pkl  # Saved trained model
│   └── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
│
├── results/
│   ├── sentiment_distribution.png
│   ├── sentiment_by_airline.png
│   ├── confusion_matrix_*.png
│   └── model_comparison.png
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── main.py                  # Main script to run the analysis
└── LICENSE                  # License file
```

## 🔍 Methodology

### 1. Data Preprocessing

- Convert text to lowercase
- Remove URLs, usernames, and special characters
- Remove stopwords (common words like "the", "and", etc.)
- Apply lemmatization to reduce words to their base form
- Tokenize the text (split into individual words)

### 2. Feature Extraction

- Convert processed text to numerical features using TF-IDF vectorization
- Create a feature matrix suitable for machine learning algorithms

### 3. Model Building

We implement and compare four different classification models:

- **Naive Bayes**: A simple probabilistic classifier based on Bayes' theorem
- **Logistic Regression**: A linear model for classification
- **Support Vector Machine (SVM)**: Effective for text classification tasks
- **Random Forest**: An ensemble learning method using multiple decision trees

### 4. Evaluation Metrics

- Accuracy: Percentage of correctly classified tweets
- Precision: Accuracy of positive predictions
- Recall: Ability to find all positive instances
- F1-score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed breakdown of classification performance

## 📊 Results

The project provides various visualizations including:

- Distribution of sentiment across all tweets
- Sentiment breakdown by airline
- Confusion matrices for each model
- Comparison of model performance

Our analysis found that:
- The majority of tweets express negative sentiment
- [Airline name] received the most negative feedback
- The [Model name] classifier achieved the highest accuracy at [X]%
- Common themes in negative tweets include: [themes]

## 🔮 Prediction Function

The project includes a function to predict sentiment for new tweets:

```python
from src.predict import predict_sentiment

new_tweets = [
    "@VirginAmerica your flight attendants are amazing!",
    "@united my flight has been delayed for 3 hours. Terrible service!"
]

predictions = predict_sentiment(new_tweets)
print(predictions)  # Output: ['positive', 'negative']
```

## 🚧 Future Improvements

- Implement deep learning models (LSTM, BERT) for potentially higher accuracy
- Create a simple web app for real-time sentiment prediction
- Expand the analysis to include more recent data
- Add topic modeling to identify specific issues in negative tweets
- Improve preprocessing for handling slang and airline-specific terminology

## 📚 References

- Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media.
- Jurafsky, D., & Martin, J. H. (2021). Speech and Language Processing. Draft.
- Kaggle. (2015). Twitter US Airline Sentiment. Retrieved from https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- [Anees-Mohamed](https://github.com/anees-mohamed-ar)


## ✉️ Contact
Insta : @anees_a_r__

For any questions or feedback, please contact [aneesmohmaed113@gmail.com].
