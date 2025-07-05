# 🎬 Sentiment Analysis on Movie Reviews

This project uses machine learning to perform **sentiment analysis** on movie reviews. The goal is to automatically classify each review as **positive** or **negative** based on its content.

---

## 📂 Project Structure

```
movie_sentiment_analysis/
│
├── app.py                   # Flask/Streamlit web app (entry point)
├── train_model.py           # Script to train the sentiment classifier
├── imdb.csv                 # Sample dataset from IMDb
├── imdb_top_1000.csv        # Top-rated IMDb movie data
├── movie_reviews_balanced.csv # Final preprocessed and balanced dataset
├── sentiment_model.pkl      # Trained sentiment classifier (logistic regression/SVM)
├── tfidf_vectorizer.pkl     # TF-IDF vectorizer used in training
├── requirements.txt         # List of Python dependencies
├── venv/                    # (ignored) Python virtual environment
```

---

## 🔍 Features

- Cleaned and balanced dataset of movie reviews
- Preprocessing with **TF-IDF vectorization**
- Trained model saved using `joblib` or `pickle`
- Optional web app for real-time sentiment prediction

---

## 🚀 How to Run the Project

### 📦 1. Clone the Repository

```bash
git clone https://github.com/TANYA2405/movie_sentiment_analysis.git
cd movie_sentiment_analysis
```

### 🐍 2. Set Up Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# OR
source venv/bin/activate  # On Mac/Linux
```

### 📥 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### ⚙️ 4. Train the Model (If needed)

```bash
python train_model.py
```

### 🌐 5. Run the Web App (Streamlit or Flask)

```bash
python app.py
```
Then open the link shown in your terminal to use the sentiment classifier!

---

## 📊 Model Info

- **Vectorization:** TF-IDF
- **Classifier:** Logistic Regression (or SVM/Naive Bayes)
- **Accuracy:** ~X% on test data (Edit this with your actual metrics)

---

## 🧠 Sample Input

**Input:**  
`The movie was a masterpiece of storytelling and visuals.`  
**Output:**  
`Positive`

---

## 🛠️ Built With

- Python
- scikit-learn
- pandas
- numpy
- Flask / Streamlit
- joblib / pickle

---

## 📎 Notes

- Large files like models and datasets are tracked via Git LFS or hosted directly in the repo.
- For faster results, consider reducing the dataset or using model quantization.

---

## 🙋‍♀️ Author

**Tanya Arya**  
🔗 [https://github.com/TANYA2405](https://github.com/TANYA2405)

---

## 📄 License

This project is licensed under the MIT License.
