#!/usr/bin/env python
# coding: utf-8

# # library imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
import re, string
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[2]:


data_og = pd.read_csv('dataset_elec_4000.csv') # read the dataset
print(data_og.shape)


# In[3]:


data_og['rating'].value_counts()


# In[4]:


print(data_og)


# In[5]:


data_og.describe()


# # preprocessing

# In[6]:


# normalization
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')


# # remove special characters

# In[7]:


#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
data_og['review'] = data_og['review'].apply(remove_special_characters)


# # text stemming

# In[8]:


# stemming the text
def stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
data_og['review']=data_og['review'].apply(stemmer)


# # stopwords setting

# In[9]:


#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#apply function on review column
data_og['review'] = data_og['review'].apply(remove_stopwords)


# # Normalized train reviews

# In[10]:


norm_train_reviews = data_og.review[:3500]
norm_train_reviews[0]


# # Normalized test reviews

# In[11]:


norm_test_reviews=data_og.review[3500:]
norm_test_reviews[3501]


# # Bags of words model

# In[12]:


#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
#transformed train reviews
cv_train_reviews=cv.fit_transform(norm_train_reviews)
#transformed test reviews
cv_test_reviews=cv.transform(norm_test_reviews)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)
#vocab=cv.get_feature_names()-toget feature names


# # Term Frequency-Inverse Document Frequency model (TFIDF)

# In[13]:


#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tv_train_reviews=tv.fit_transform(norm_train_reviews)
#transformed test reviews
tv_test_reviews=tv.transform(norm_test_reviews)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)


# In[14]:


#labeling the sentient data
lb = LabelBinarizer()
#transformed sentiment data
rating_data = lb.fit_transform(data_og['rating'])
print(rating_data.shape)


# In[15]:


# spliting the sentiment data
train_rating = rating_data[:3500]
test_rating = rating_data[3500:]
print(train_rating.shape)
print(test_rating.shape)


# In[16]:


train_rating = np.reshape(train_rating,(-1))
test_rating = np.reshape(test_rating,(-1))
print( train_rating.shape )             # for debug 
print( train_rating.ndim )  


# In[17]:


# training the model
mnb = MultinomialNB()
# fitting the svm for bag of words
mnb_bow = mnb.fit(cv_train_reviews, train_rating)
print(mnb_bow)
# fitting the svm for tfidf features
mnb_tfidf = mnb.fit(tv_train_reviews, train_rating)
print(mnb_tfidf)


# In[18]:


# predict the model for bag of words
mnb_bow_predict = mnb.predict(cv_test_reviews)
print(mnb_bow_predict)
# predict the model for tfidf features
mnb_tfidf_predict=mnb.predict(tv_test_reviews)
print(mnb_tfidf_predict)


# In[19]:


#Accuracy score for bag of words
mnb_bow_score=accuracy_score(test_rating, mnb_bow_predict)
print("mnb_bow_score :",mnb_bow_score)
#Accuracy score for tfidf features
mnb_tfidf_score=accuracy_score(test_rating, mnb_tfidf_predict)
print("mnb_tfidf_score :",mnb_tfidf_score)


# In[20]:


#Classification report for bag of words 
mnb_bow_report=classification_report(test_rating,mnb_bow_predict,target_names=['Positive','Negative'])
print(mnb_bow_report)
#Classification report for tfidf features
mnb_tfidf_report=classification_report(test_rating,mnb_tfidf_predict,target_names=['Positive','Negative'])
print(mnb_tfidf_report)


# In[21]:


#confusion matrix for bag of words
cm_bow=confusion_matrix(test_rating,mnb_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_rating,mnb_tfidf_predict,labels=[1,0])
print(cm_tfidf)


# In[24]:


#word cloud for positive review words
plt.figure(figsize=(10,10))
positive_text=norm_train_reviews[1]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
positive_words=WC.generate(positive_text)
plt.imshow(positive_words,interpolation='bilinear')
plt.show
print(norm_train_reviews[1])


# In[28]:


#Word cloud for negative review words
plt.figure(figsize=(10,10))
negative_text=norm_train_reviews[8]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
negative_words=WC.generate(negative_text)
plt.imshow(negative_words,interpolation='bilinear')
plt.show
print(norm_train_reviews[8])


# In[ ]:




