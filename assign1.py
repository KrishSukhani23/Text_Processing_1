import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import contractions

import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

data = pd.read_csv('amazon_reviews_us_Beauty_v1_00.tsv', on_bad_lines='skip', sep='\t', low_memory=False)
data = data[['review_body', 'star_rating']]
data = data.dropna()

data['star_rating'] = np.where(data['star_rating'] == '1', 'class1', data['star_rating'])
data['star_rating'] = np.where(data['star_rating'] == '2', 'class1', data['star_rating'])
data['star_rating'] = np.where(data['star_rating'] == '3', 'class2', data['star_rating'])
data['star_rating'] = np.where(data['star_rating'] == '4', 'class3', data['star_rating'])
data['star_rating'] = np.where(data['star_rating'] == '5', 'class3', data['star_rating'])

data_class1 = data[data['star_rating'] == 'class1'].sample(n=20000, replace = False)
data_class2 = data[data['star_rating'] == 'class2'].sample(n=20000, replace = False)
data_class3 = data[data['star_rating'] == 'class3'].sample(n=20000, replace = False)

df_new = pd.concat([data_class1,data_class2,data_class3])

# Cleaning

# print("Average length before cleaning : ", df_new['review_body'].apply(len).mean())
bef_cln = df_new['review_body'].apply(len).mean()

df_new['review_body'] = df_new['review_body'].str.lower()

CLEANR = re.compile('<.*?>') 
def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

df_new['review_body'] = df_new['review_body'].apply(cleanhtml)

df_new['star_rating'] = df_new['star_rating'].astype(str)
df_new['review_body'] = df_new['review_body'].astype(str)
df_new['review_body'] = df_new['review_body'].apply(cleanhtml)

df_new['review_body'] = df_new['review_body'].apply(contractions.fix)
df_new['review_body'] = df_new['review_body'].str.replace('[^A-Za-z ]', '', regex=True)
df_new['review_body'] = df_new['review_body'].str.replace('  ', ' ')

# print("Average length after cleaning : ", df_new['review_body'].apply(len).mean())
aft_cln = df_new['review_body'].apply(len).mean()

print(str(bef_cln) +", " + str(aft_cln))
#Preprocessing
bef_pre = df_new['review_body'].apply(len).mean()
# print("Average length before preprocessing : ", df_new['review_body'].apply(len).mean())

stop = stopwords.words('english')
df_new['review_body_without_stopwords'] = df_new['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


from nltk.stem import WordNetLemmatizer
lemmatize_model = WordNetLemmatizer()
def lemmatize_words(text):
    words = text.split()
    words = [lemmatize_model.lemmatize(word, pos='v') for word in words]
    return ' '.join(words)

df_new['review_body_lemmatized'] = df_new['review_body'].apply(lemmatize_words)
df_new['review_body_without_stopwords_lemmatized'] = df_new['review_body_without_stopwords'].apply(lemmatize_words)

# print("Average length after preprocessing(with stopwords) : ", df_new['review_body_lemmatized'].apply(len).mean())
# print("Average length after preprocessing(without stopwords) : ", df_new['review_body_without_stopwords_lemmatized'].apply(len).mean())

aft_pre = df_new['review_body_without_stopwords_lemmatized'].apply(len).mean()
print(str(bef_pre) +", " + str(aft_pre))
#TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_new["review_body_lemmatized"])

y = df_new['star_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Perceptron

from sklearn.linear_model import Perceptron

model = Perceptron(tol=1e-3, random_state=23)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))
precision,recall,fscore,support=score(y_test,y_pred)
pr = 0
re = 0
f1 = 0
for i in range(3):
  # print('class' + str(i+1) + " Precision : " + str(precision[i]) + " Recall : " + str(recall[i]) + " F-score : " + str(fscore[i]))
  print(str(precision[i]) + " ," + str(recall[i]) + " ," + str(fscore[i]))
  pr += precision[i]
  re += recall[i]
  f1 += fscore[i]

# print("Avg Precision : " + str(pr/3) + " Avg Recall : " + str(re/3) + " Avg F-score : " + str(f1/3))
print(str(pr/3) + " ," + str(re/3) + " ," + str(f1/3))

#SVM

from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=23, tol=1e-5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
precision,recall,fscore,support=score(y_test,y_pred)
pr = 0
re = 0
f1 = 0
for i in range(3):
  # print('class' + str(i+1) + " Precision : " + str(precision[i]) + " Recall : " + str(recall[i]) + " F-score : " + str(fscore[i]))
  print(str(precision[i]) + " ," + str(recall[i]) + " ," + str(fscore[i]))
  pr += precision[i]
  re += recall[i]
  f1 += fscore[i]

# print("Avg Precision : " + str(pr/3) + " Avg Recall : " + str(re/3) + " Avg F-score : " + str(f1/3))
print(str(pr/3) + " ," + str(re/3) + " ," + str(f1/3))

#Logistic Regression

from sklearn.linear_model import LogisticRegression

clf= LogisticRegression(random_state=23, max_iter=100000)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
# print(classification_report(y_test, y_pred))
precision,recall,fscore,support=score(y_test,y_pred)
pr = 0
re = 0
f1 = 0
for i in range(3):
  # print('class' + str(i+1) + " Precision : " + str(precision[i]) + " Recall : " + str(recall[i]) + " F-score : " + str(fscore[i]))
  print(str(precision[i]) + " ," + str(recall[i]) + " ," + str(fscore[i]))
  pr += precision[i]
  re += recall[i]
  f1 += fscore[i]

# print("Avg Precision : " + str(pr/3) + " Avg Recall : " + str(re/3) + " Avg F-score : " + str(f1/3))
print(str(pr/3) + " ," + str(re/3) + " ," + str(f1/3))

#Naive Bayes

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# print(classification_report(y_test, y_pred))
precision,recall,fscore,support=score(y_test,y_pred)
pr = 0
re = 0
f1 = 0
for i in range(3):
  # print('class' + str(i+1) + " Precision : " + str(precision[i]) + " Recall : " + str(recall[i]) + " F-score : " + str(fscore[i]))
  print(str(precision[i]) + " ," + str(recall[i]) + " ," + str(fscore[i]))
  pr += precision[i]
  re += recall[i]
  f1 += fscore[i]

# print("Avg Precision : " + str(pr/3) + " Avg Recall : " + str(re/3) + " Avg F-score : " + str(f1/3))
print(str(pr/3) + " ," + str(re/3) + " ," + str(f1/3))