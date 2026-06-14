Sentiment Analysis of Twitter Data Using NLP and Machine Learning

This project aims to analyze and classify the sentiment of Twitter posts into Positive, Negative, and Neutral categories using Natural Language Processing (NLP) and Machine Learning techniques. The dataset contains over 162,000 tweets, which are processed and transformed into numerical features using TF-IDF Vectorization. A Multinomial Naive Bayes classifier is trained to predict sentiment based on tweet content. The model's performance is evaluated using Accuracy Score, Classification Report, and Confusion Matrix. Data visualization techniques are used to present sentiment distribution and model results effectively. This project demonstrates the practical application of NLP for understanding public opinion, social media trends, and customer sentiment analysis.
Dataset


The project uses the Twitter Data dataset containing 162,980 tweets with sentiment labels. The dataset includes preprocessed tweet text and corresponding sentiment categories.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
NLTK
Matplotlib
Seaborn
Methodology
Loaded and cleaned the Twitter dataset.
Removed missing values and prepared text data for analysis.
Converted textual data into numerical features using TF-IDF Vectorization.
Split the dataset into training and testing sets.
Trained a Multinomial Naive Bayes classifier for sentiment prediction.
Evaluated model performance using Accuracy Score, Classification Report, and Confusion Matrix.
Visualized prediction results using heatmaps and graphical representations.
Key Features
Text preprocessing and cleaning
Sentiment classification using Machine Learning
TF-IDF feature extraction
Model training and evaluation
Confusion Matrix visualization
Large-scale Twitter sentiment analysis
Results

The model successfully classified Twitter posts into different sentiment categories and demonstrated effective performance in understanding public sentiment. The Confusion Matrix and evaluation metrics were used to measure prediction accuracy and model reliability.
