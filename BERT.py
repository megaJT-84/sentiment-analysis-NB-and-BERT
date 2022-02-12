#!/usr/bin/env python
# coding: utf-8

# In[1]:


# general use libraries
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
import string
import re

# preprocessing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# BERT libraries 
from transformers import TFBertModel, BertConfig, BertTokenizerFast, TFAutoModel

# Then what you need from tensorflow.keras
import tensorflow as tsf
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, SpatialDropout1D, Conv1D, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import LAMB, AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical


# In[2]:


data_og = pd.read_csv('dataset_elec_4000.csv')


# In[3]:


data_og.head()


# In[4]:


data_og.info()


# In[5]:


target_categories = ["0","1"]


# # preprocessing 

# In[6]:


# the BERT model to use
model_name = 'bert-base-cased'

# max length of tokens
length = len(data_og.review)
dff = [len(i.split(" ")) for i in data_og.review[:length]]
max_length = max(dff)+3

# transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False


# In[7]:


# BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)# setting English stopwords
stopwords_list = nltk.corpus.stopwords.words('english')

# function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text = re.sub(pattern,'',text)
    return text
# apply function on review column
data_og['review'] = data_og['review'].apply(remove_special_characters)

#removing the stopwords
def remove_stopwords(text):
    text_tokens = text.split(" ")
    text_tokens_filtered= [word for word in text_tokens if not word in stopwords_list]
    return (" ").join(text_tokens_filtered)
#apply function on review column
data_og['review'] = data_og['review'].apply(remove_stopwords)


# # cleaned subsets

# In[8]:


# train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data_og.index.values, data_og.rating.values, test_size=0.1, random_state=42, stratify=data_og.rating)

# train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

data_og['data_type'] = ['not_set']*data_og.shape[0]
data_og.loc[X_train, 'data_type'] = 'train'
data_og.loc[X_val, 'data_type'] = 'val'
data_og.loc[X_test, 'data_type'] = 'test'

data_divided = data_og.dropna()
print(data_divided)


# # word clouds display

# In[9]:


# negative words
negative = data_og[data_og['rating'] == 0]
wordCloud = WordCloud(background_color="white", width=1600, height=800).generate(' '.join(negative.review))
plt.figure(figsize=(20,10), facecolor='k')
plt.imshow(wordCloud)


# In[10]:


# positive words
positive = data_og[data_og['rating'] == 1]
wordCloud = WordCloud(background_color="white", width=1600, height=800).generate(' '.join(positive.review))
plt.figure(figsize=(20,10), facecolor='k')
plt.imshow(wordCloud)


# In[11]:


y_rating = to_categorical(data_divided[data_divided.data_type=='train'].rating)
x = tokenizer(
    text=data_divided[data_divided.data_type=='train'].review.to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

train_set = tsf.data.Dataset.from_tensor_slices((x['input_ids'], x['attention_mask'], y_rating))
def map_func(input_ids, masks, labels):
    # convert three-item tuple into a two-item tuple where the input item is a dictionary
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

train_set = train_set.map(map_func)
batch_size = 32

# shuffle and batch - dropping any remaining samples that don't cleanly
train_set = train_set.shuffle(100).batch(batch_size, drop_remainder=True)

train_set.take(1)


# In[12]:


y_rating = to_categorical(data_divided[data_divided.data_type=='val'].rating)

# Tokenize the input 
x = tokenizer(
    text=data_divided[data_divided.data_type=='val'].review.to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

val = tsf.data.Dataset.from_tensor_slices((x['input_ids'], x['attention_mask'], y_rating))
val = val.map(map_func)
val = val.shuffle(100).batch(batch_size, drop_remainder=True)


# # build model with transfer learning

# In[13]:


# build model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

bert = TFAutoModel.from_pretrained('bert-base-cased')
embeddings = bert.bert(inputs)[1]

# convert bert embeddings into 2 output classes
output = Flatten()(embeddings)
output = Dense(64, activation='relu')(output)
output = Dense(32, activation='relu')(output)

output = Dense(2, activation='softmax', name='outputs')(output)

model = Model(inputs=inputs, outputs=output)

# Take a look at the model
model.summary()


# In[14]:


optimizer = AdamW(learning_rate=1e-5, weight_decay=1e-6)
loss = CategoricalCrossentropy()
acc = CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])


# # train BERT model with training and validation sets

# In[ ]:


# fit the model
history = model.fit(
    train_set,
    validation_data=val,
    epochs=1)


# In[ ]:


model.save_weights('./sentiment-analysis-on-movie-reviews/bert_weights.h5')


# In[ ]:


model.load_weights('./sentiment-analysis-on-movie-reviews/bert_weights.h5')


# # confusion matrix

# In[ ]:


def map_func(input_ids, masks):
    return {'input_ids': input_ids, 'attention_mask': masks}

# tokenize input
x = tokenizer(
    text=data_divided[data_divided.data_type=='test'].review.to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

test = tsf.data_divided.Dataset.from_tensor_slices((x['input_ids'], x['attention_mask']))
test = test.map(map_func)
test = test.batch(32)


# In[ ]:


y_test = data[data.data_type=='test'].Sentiment
y_pred = model.predict(test).argmax(axis=-1)


# In[ ]:


# plot confusion matrix for test data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

print("BERT Train Accuracy Score :      {:.0f}% ".format(history.history['accuracy'][-1]*100))
print("BERT Validation Accuracy Score : {:.0f}% ".format(history.history['val_accuracy'][-1]*100))
print("BERT Test Accuracy Score  :      {:.0f}% ".format(accuracy_score(y_test, y_pred)*100))
print()
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
disp.plot()

# classification report for validation data
print(classification_report(y_test, y_pred, target_names=target_categories))


# In[ ]:




