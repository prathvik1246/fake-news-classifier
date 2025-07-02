# 🧠 Fake News Classifier

A machine learning project that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) and supervised learning.

This project compares multiple models and uses TF-IDF text vectorization, logistic regression, and wordcloud visualizations. It includes a CLI tool for real-time classification.

---

## 🚀 Features

- ✅ Preprocesses text using a custom cleaning pipeline  
- ✅ Compares Logistic Regression, Naive Bayes, and Random Forest  
- ✅ Saves the best model + vectorizer for reuse  
- ✅ CLI tool to classify custom news input with confidence score  
- ✅ WordClouds for Real vs Fake article insights  
- ✅ Modular codebase with reusable components

---

## 🗂 Project Structure

fake-news-classifier/
├── data/                 # Raw dataset (Fake.csv, True.csv)
├── models/               # Saved model and vectorizer
├── src/
│   ├── train.py          # Train multiple models and save best
│   ├── predict.py        # CLI tool for classifying news input
│   ├── utils.py          # Text preprocessing functions
│   └── wordclouds.py     # Generates word clouds for real/fake news
├── requirements.txt
└── README.md

---

## 📦 Installation

### 1. Clone the repo

git clone https://github.com/prathvik1246/fake-news-classifier.git
cd fake-news-classifier

### 2. Install dependencies

pip install -r requirements.txt

If you don’t have a `requirements.txt`, you can use:

pandas
scikit-learn
nltk
wordcloud
matplotlib

---

## 🧪 Train the Model

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset  
2. Place `Fake.csv` and `True.csv` inside the `data/` folder

Then run:

python3 src/train.py

This will:
- Clean and vectorize the text  
- Compare 3 machine learning models  
- Save the best model to `models/logistic_model.pkl`

---

## 💬 Predict News (CLI)

To classify a custom news article:

python3 src/predict.py --text "NASA finds water on Mars!"

Example Output:

Prediction: Real  
Confidence: 94.23%

---

## 📊 WordClouds

To visualize word frequency in fake vs real news:

python3 src/wordclouds.py

This will display:
- A Fake News WordCloud  
- A Real News WordCloud

---

## 📈 Sample Accuracy

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| ~92.6%   |
| Naive Bayes        | ~91.1%   |
| Random Forest      | ~89.7%   |

---

## 🧠 Technologies Used

- Python  
- scikit-learn  
- nltk  
- wordcloud  
- matplotlib  
- pandas  

---

## 📚 Dataset

Kaggle - Fake and Real News Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## 📜 License

This project is open source and available under the MIT License.

---

## ✨ Author

Prathvik Potla – https://github.com/prathvik1246
