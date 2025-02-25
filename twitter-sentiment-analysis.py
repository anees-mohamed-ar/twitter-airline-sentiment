# Twitter Airline Sentiment Analysis
# A complete implementation for beginners

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Data Loading and Exploration
print("1. Loading and exploring the dataset...")
# Load the dataset
df = pd.read_csv('Tweets.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Basic statistics about the dataset
print("\nSentiment distribution:")
print(df['airline_sentiment'].value_counts())

# 2. Data Visualization
print("\n2. Visualizing the data...")

# Sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='airline_sentiment', data=df)
plt.title('Distribution of Sentiment')
plt.savefig('sentiment_distribution.png')

# Sentiment distribution by airline
plt.figure(figsize=(12, 6))
sns.countplot(x='airline', hue='airline_sentiment', data=df)
plt.title('Sentiment Distribution by Airline')
plt.xticks(rotation=45)
plt.savefig('sentiment_by_airline.png')

# 3. Text Preprocessing
print("\n3. Preprocessing text data...")

# Define a function to clean and preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove usernames
    text = re.sub(r'@\w+', '', text)
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

# Apply preprocessing to the tweet text
df['processed_text'] = df['text'].apply(preprocess_text)

print("Sample processed tweets:")
for i in range(3):
    print(f"Original: {df['text'].iloc[i]}")
    print(f"Processed: {df['processed_text'].iloc[i]}")
    print()

# 4. Feature Extraction
print("\n4. Converting text to numerical features...")

# Using TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['processed_text'])
y = df['airline_sentiment']

print(f"Feature matrix shape: {X.shape}")

# 5. Train-Test Split
print("\n5. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 6. Model Building and Evaluation
print("\n6. Building and evaluating models...")

# Define a function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = accuracy*100
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, conf_matrix, y_pred

# Initialize models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': LinearSVC(dual=False),
    'Random Forest': RandomForestClassifier()
}

# Evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    accuracy, report, conf_matrix, predictions = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {
        'accuracy': accuracy,
        'report': report,
        'conf_matrix': conf_matrix,
        'predictions': predictions
    }
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:\n{report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y.unique()),
                yticklabels=sorted(y.unique()))
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')

# 7. Model Comparison
print("\n7. Comparing model performances...")

# Compare accuracies
accuracies = [results[model]['accuracy'] for model in models]
model_names = list(models.keys())

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies)
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.savefig('model_comparison.png')

# 8. Best Model Analysis
print("\n8. Analyzing the best model...")

# Find the best model
best_model_name = model_names[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# 9. Error Analysis
print("\n9. Performing error analysis...")

# Get predictions from the best model
best_predictions = results[best_model_name]['predictions']

# Create a DataFrame with actual and predicted values
error_df = pd.DataFrame({
    'text': df['text'].iloc[y_test.index],
    'processed_text': df['processed_text'].iloc[y_test.index],
    'actual': y_test.values,
    'predicted': best_predictions
})

# Filter for incorrect predictions
incorrect = error_df[error_df['actual'] != error_df['predicted']]
print(f"Number of incorrectly classified tweets: {len(incorrect)}")

# Sample some misclassified examples
print("\nSample misclassifications:")
for i in range(min(5, len(incorrect))):
    print(f"Text: {incorrect['text'].iloc[i]}")
    print(f"Actual: {incorrect['actual'].iloc[i]}, Predicted: {incorrect['predicted'].iloc[i]}")
    print()

# 10. Save the best model for future use
print("\n10. Saving the best model for future use...")
import pickle

# Get the best model
best_model = models[best_model_name]

# Save the model and vectorizer
with open('best_sentiment_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Model and vectorizer saved to disk!")

# 11. Function for predicting sentiment of new tweets
print("\n11. Creating a function for sentiment prediction...")

def predict_sentiment(new_tweets, model_file='best_sentiment_model.pkl', vectorizer_file='tfidf_vectorizer.pkl'):
    """
    Predict sentiment for new tweets using the saved model.
    
    Parameters:
    -----------
    new_tweets : list of str
        List of tweet texts to predict sentiment for
    model_file : str
        Path to the saved model file
    vectorizer_file : str
        Path to the saved vectorizer file
        
    Returns:
    --------
    predictions : list
        List of sentiment predictions
    """
    # Load model and vectorizer
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Preprocess the new tweets
    processed_tweets = [preprocess_text(tweet) for tweet in new_tweets]
    
    # Transform using the loaded vectorizer
    X_new = vectorizer.transform(processed_tweets)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    return predictions

# Example usage of prediction function
sample_tweets = [
    "@VirginAmerica plus you guys fly to the west coast, which @AmericanAir doesn't...",
    "@united your customer service is horrible. I will never fly with you again!",
    "Thanks @JetBlue for the amazing flight experience as always!"
]

predictions = predict_sentiment(sample_tweets)
for tweet, pred in zip(sample_tweets, predictions):
    print(f"Tweet: {tweet}")
    print(f"Predicted sentiment: {pred}\n")

print("completed successfully!")