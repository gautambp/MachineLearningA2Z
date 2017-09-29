# -*- coding: utf-8 -*-

# read the dataset
import pandas as pd
# use tab delimiter and ignore quotes around text
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
review_count, cols = dataset.shape

# clean the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
cleaned_reviews = []
for i in range(0, review_count):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    # use downloaded english words to filter out the rest; this also removes a, an, the, and, this,...
    eng_words = set(stopwords.words('english'))
    # Perform stemming as well.. e.g., love, lovely, loved, loving,.. = love (utilize root word)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in eng_words]
    # join individual words in each review back into sentence
    review = ' '.join(review)
    cleaned_reviews.append(review)

# create bag of words model
# one row for each review.. picks all words (or max features) used across reviews and makes them column
# cell value of 1 indicates presence of the word (indicated as column name) in the current review
from sklearn.feature_extraction.text import CountVectorizer
# only 1500 columns of most frequently used words
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(cleaned_reviews).toarray()
y = dataset.iloc[:, 1].values

# now use naive bayes model to train and predict

# split dataset into train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5, random_state=0)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)

# predict on the test data and learn about performance using confusion matrix
y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
