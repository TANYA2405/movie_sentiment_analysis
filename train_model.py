import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load labeled dataset
try:
    df = pd.read_csv("movie_reviews_balanced.csv")  # Make sure this file is in the same directory
    print("Dataset loaded successfully.")
    print(f"Number of rows loaded: {len(df)}")
    print(f"Columns in the loaded DataFrame: {df.columns.tolist()}")

    # Check for common alternative column names and rename them
    if 'review' in df.columns and 'sentiment' in df.columns:
        df.rename(columns={'review': 'Review', 'sentiment': 'Sentiment'}, inplace=True)
        print("Renamed 'review' to 'Review' and 'sentiment' to 'Sentiment'.")
    elif 'text' in df.columns and 'label' in df.columns:
        df.rename(columns={'text': 'Review', 'label': 'Sentiment'}, inplace=True)
        print("Renamed 'text' to 'Review' and 'label' to 'Sentiment'.")

    # Verify columns after potential renaming
    if 'Review' not in df.columns:
        raise KeyError("Column 'Review' not found in the DataFrame after attempting rename. Please check your CSV file's column names.")
    if 'Sentiment' not in df.columns:
        raise KeyError("Column 'Sentiment' not found in the DataFrame after attempting rename. Please check your CSV file's column names.")


except FileNotFoundError:
    print("Error: 'movie_reviews_balanced.csv' not found. Please ensure the file is in the same directory as train_model.py")
    exit()
except KeyError as e:
    print(f"\nError: {e}. It seems your CSV file does not contain the expected 'Review' and 'Sentiment' columns.")
    print("Please open 'movie_reviews_balanced.csv' in a text editor or spreadsheet and verify the exact column names.")
    print("Expected: 'Review' and 'Sentiment' (case-sensitive).")
    print(f"Actual columns found: {df.columns.tolist() if 'df' in locals() else 'DataFrame not loaded'}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the dataset: {e}")
    exit()

# Clean text function
def clean_text(text):
    text = str(text) # Ensure text is always a string
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# Convert Review to string and clean it
df["Review"] = df["Review"].astype(str).fillna("")
df["cleaned_review"] = df["Review"].apply(clean_text)

print("\n--- Debugging: Checking cleaned reviews ---")
print(f"Original number of reviews: {len(df)}")
print("\nSample of cleaned reviews before dropping empty ones (first 10):")
print(df["cleaned_review"].head(10))
print(f"Number of completely empty strings before drop: {(df['cleaned_review'].str.strip() == '').sum()}")

# Drop empty cleaned reviews
initial_rows = len(df)
df = df[df["cleaned_review"].str.strip() != ""]
rows_dropped = initial_rows - len(df)
print(f"\nDropped {rows_dropped} rows due to empty cleaned reviews.")
print(f"Number of reviews after dropping empty ones: {len(df)}")

print("\nSample of cleaned reviews AFTER dropping empty ones (first 10):")
print(df["cleaned_review"].head(10))
print(f"Are there any remaining completely empty strings after drop? {(df['cleaned_review'].str.strip() == '').any()}")

# Check the length of cleaned reviews to understand remaining content
df['cleaned_review_length'] = df['cleaned_review'].apply(len)
print("\nStatistics of cleaned review lengths:")
print(df['cleaned_review_length'].describe())

print("\nShortest 20 cleaned reviews (up to 20 characters):")
print(df[df['cleaned_review_length'] <= 20]['cleaned_review'].head(20))

# If after all this, df is empty, then there's no data to process
if df.empty:
    print("\nError: After cleaning and dropping empty rows, the DataFrame is empty. ")
    print("This means your 'movie_reviews_balanced.csv' likely contains no valid text reviews ")
    print("that pass the 'clean_text' filter, or all reviews became empty strings.")
    print("Please check your 'movie_reviews_balanced.csv' file content carefully or relax the cleaning.")
    exit()

# Encode sentiment (Positive = 1, Negative = 0)
# ***THIS IS THE CHANGED LINE***
# Changed from {'Positive': 1, 'Negative': 0} to {'positive': 1, 'negative': 0}
df['Sentiment'] = df['Sentiment'].astype(str).map({'positive': 1, 'negative': 0})

# Drop rows with missing sentiment (after mapping, NaN might appear if values were not 'positive'/'negative')
df = df.dropna(subset=["Sentiment"])
print(f"\nNumber of reviews after dropping missing sentiment: {len(df)}")

if df.empty:
    print("\nError: After encoding sentiment and dropping missing values, the DataFrame is empty.")
    print("Ensure your 'Sentiment' column strictly contains 'positive' or 'negative' values (case-sensitive).")
    exit()


# Vectorization using TF-IDF
# Key change: min_df=1 to ensure even words appearing once are included
tfidf = TfidfVectorizer(max_features=5000, min_df=1)
print(f"\nInitializing TfidfVectorizer with max_features={tfidf.max_features}, min_df={tfidf.min_df}")

try:
    df["cleaned_review"] = df["cleaned_review"].fillna("")
    X = tfidf.fit_transform(df['cleaned_review']).toarray()
    print(f"TF-IDF vectorization complete. Vocabulary size: {len(tfidf.vocabulary_)}")
    if len(tfidf.vocabulary_) == 0:
        print("Warning: Vocabulary is still empty despite min_df=1. This indicates very unusual data (e.g., all reviews identical after cleaning).")
        print("Please check your 'cleaned_review' column contents from the debug prints above.")
        exit() # Exit if vocabulary is empty to prevent further errors
except ValueError as e:
    print(f"\nError during TF-IDF vectorization: {e}")
    print("This often means that even with min_df=1, no unique tokens could be formed.")
    print("Double-check your 'cleaned_review' column contents from the debug prints above.")
    exit()

y = df['Sentiment']

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)
print(f"Shape after SMOTE: X={X.shape}, y={y.shape}")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train/Test split: X_train={X_train.shape}, X_test={X_test.shape}")

# Train the model
model = RandomForestClassifier(random_state=42)
print("Training RandomForestClassifier...")
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
try:
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("\nModel and vectorizer saved successfully (sentiment_model.pkl, tfidf_vectorizer.pkl).")
except Exception as e:
    print(f"\nError saving model/vectorizer: {e}")