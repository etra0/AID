
# coding: utf-8

# # TMDb y redes neuronales

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as r
import json
r.seed(1234)


# Lo primero que haremos, será importar la data ([Source](https://www.kaggle.com/tmdb/tmdb-movie-metadata/data)) usando pandas

# In[23]:


df = pd.read_csv("tmdb_5000_movies.csv")


# In[24]:


df.head()


# In[25]:


df.info()


# En este caso, lo que necesitamos es sólamente el **género** y la **overview** para entrenar la red neuronal, por lo tanto se extraerá ésto.

# In[26]:


print(df['genres'][0])
print(df['overview'][0])


# Se puede observar, como ser humano hay una relación inherente entre las etiquetas y la descripción de la película. En nuestro clasificador de texto solo se puede asignar una etiqueta al texto. Suponemos que todas las etiquetas son representativas, así que se escogerá la etiqueta *categorizadora* de forma aleatoria.

# In[27]:


def generate_data():
    final_list = []
    for i, row in df.iterrows():
        temp_dict = dict()
        genres = json.loads(row['genres'])
        
        # peliculas sin genero
        if len(genres) == 0 or (type(row['overview']) == str and len(row['overview']) < 3) or type(row['overview']) == float:
            continue
            
        selected_genre = r.choice(genres)
        temp_dict['class'] = selected_genre['name']
        temp_dict['sentence'] = row['overview']
        final_list.append(temp_dict)
    return final_list


# In[28]:


training_data = generate_data()


# In[29]:


r.choice(training_data)


# Una vez preparado la `training_data`, se procederá a construir la red neuronal.

# In[30]:


# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
stemmer = LancasterStemmer()


# In[31]:


words = []
classes = []
documents = []
ignore_words = ['?', '.', ',', "'", '"']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "classes", classes)


# In[32]:


# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w])
print (training[i])
print (output[i])


# In[12]:


X = np.array(training)
y = np.array(output)


# In[13]:


train_X = X[:int(len(X)*0.8)]
train_Y = y[:int(len(y)*0.8)]

test_X = X[int(len(X)*0.8):]
test_Y = y[int(len(y)*0.8):]


# In[14]:


print(train_X.shape)
print(test_X.shape)


# In[15]:


print(classes)


# In[34]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(1024, input_dim=train_X.shape[1], kernel_initializer='uniform', activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1024, kernel_initializer='uniform', activation="relu"))
model.add(Dropout(0.1))
#model.add(Dense(512, kernel_initializer='uniform', activation="relu"))
model.add(Dense(train_Y.shape[1]))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[35]:


model.fit(train_X, train_Y, epochs=5, batch_size=256, verbose=1, shuffle=True)


# In[36]:


score=model.evaluate(test_X, test_Y, verbose=1)
print("\nLoss: %.3f \t Accuracy: %.3f" % (score[0], score[1]))


# In[37]:


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
                    
    return(np.array(bag))


# In[38]:


def predict(t, threshold=0.08):
    bag = bow(t, words)
    bag = np.array(bag).reshape(1, 15007)
    l = sorted(zip(classes, model.predict(bag)[0]), key=lambda x: x[1], reverse=True)
    return [i for i in l if i[1] > threshold]


# In[39]:


test = ["A family goes out for vacations with their kids to enjoy the life", 
        "A bat have to save the world against superman",
       "An Alien arrived to my home and tried to kill me",
       "A haunted house is in front of my window",
       "A police man is killing someone"]

for t in test:
    print(t, predict(t))
    print()

