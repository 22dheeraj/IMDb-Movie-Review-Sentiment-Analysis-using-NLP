🎬 IMDb Movie Review Sentiment Analysis

A Natural Language Processing (NLP) Project

📖 Overview

This project applies Natural Language Processing (NLP) and Machine Learning to classify IMDb movie reviews as positive or negative.

By using text preprocessing techniques, feature engineering (TF-IDF, embeddings), and classification algorithms, the model is trained to predict the sentiment of reviews with high accuracy.

🔎 Use Case: Helps platforms like IMDb, movie producers, and critics to understand public opinion, improve recommendations, and design better marketing strategies.

❓ Problem Statement

The main objective is to build a classification model that predicts the sentiment (positive/negative) of IMDb reviews.

Approach

Data Exploration & Cleaning → Balanced dataset of 50,000 reviews (25k positive, 25k negative).

Preprocessing → Remove stopwords, punctuation, lowercase text, tokenize, lemmatize.

Feature Engineering → TF-IDF vectorization + additional textual features.

Model Development → Train multiple classifiers (Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost).

Hyperparameter Tuning → Optimize models using GridSearchCV.

📂 Dataset Information

The IMDb dataset contains user-submitted reviews labeled with sentiment.

Field	Description
review	Text of the movie review
sentiment	Positive / Negative

📌 Dataset Size: 50,000 reviews (balanced)

📝 Project Tasks
🔹 Task 1: Data Exploration & Preprocessing (5 Marks)

Checked dataset shape, missing values, class balance.

Explored review lengths distribution.

Performed text cleaning using NLTK:

Lowercasing

Removing HTML tags, punctuation, and special characters

Removing stopwords

Tokenization

Lemmatization

🔹 Task 2: Feature Engineering (10 Marks)

TF-IDF Vectorization for text-to-numeric transformation.

Additional textual features:

Word count

Character count

Average word length

📝 Project Tasks
🔹 Task 1: Data Exploration & Preprocessing 

Checked dataset shape, missing values, class balance.

Explored review lengths distribution.

Performed text cleaning using NLTK:

Lowercasing

Removing HTML tags, punctuation, and special characters

Removing stopwords

Tokenization

Lemmatization

🔹 Task 2: Feature Engineering 

TF-IDF Vectorization for text-to-numeric transformation.

Additional textual features:

Word count

Character count

Average word length

🔹 Task 3: Model Development

Built and evaluated multiple classification models:

Logistic Regression

Naive Bayes (MultinomialNB)

Support Vector Machine (LinearSVC)

Random Forest

XGBoost

✔️ Hyperparameter tuning performed using GridSearchCV.

🔹 Task 4: Model Evaluation (5 Marks)

Metrics used: Accuracy, Precision, Recall, F1-score

Confusion Matrices plotted

Final Evaluation Table created before & after tuning

Evaluation → Compare performance using accuracy, precision, recall, F1-score.

⚙️ Tools & Libraries

Python 

Data Analysis & ML: pandas, numpy, scikit-learn, xgboost, Random Forest, Naive Bayes, SVM, Logistic Regression

NLP: NLTK, spaCy

Visualization: matplotlib, seaborn, wordcloud

Environment: Jupyter Notebook

📊 Deliverables

✔️ Jupyter Notebook with code & analysis
✔️ Preprocessed dataset & feature representations
✔️ Trained models with evaluation results
✔️ Visualizations (confusion matrices, word clouds, plots)


