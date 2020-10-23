#api for extracting the results from wikidata

#https://www.wikidata.org/w/api.php?search=las&language=en&uselang=en&format=jsonfm&limit=25&action=wbsearchentities

# importing modules 
import requests 
from lxml import etree 
import wikipedia
import sys
import re
import pickle
import numpy as np
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib as plt
from tensorflow import keras
from nltk.corpus import stopwords 
from nltk.corpus import wordnet,words
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

SEARCHPAGE = str(sys.argv[1])

page=wikipedia.WikipediaPage(SEARCHPAGE)
content=page.content

content_list=content.split('.')

#for i in content_list:
#    print(i) 

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


max_words = 50
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)

#defining the text fields from training set as train_text and similiarly test_text
train_text= pd.DataFrame({'words':content_list})
#print(train_text)

#print("###################################   TRAINING DATASET DESCRIPTION  ###############################################################")
#print(train_text.describe())

#remove unwanted from the questions
#query = 'What is Nahuatl word for tomato and how did Aztecs called tomato ?'
query=str(sys.argv[2])
stopperwords = ['what','where','when','who','which','whom','whose','why','how','?']
querywords = query.split()

resultwords  = [word for word in querywords if word.lower() not in stopperwords]
result = ' '.join(resultwords)
result=result.replace('?','')
#print(result)

stop_words = set(stopwords.words('english')) 
word_tokens = result.split(' ') 
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
filtered_sentence = [] 
for w in word_tokens: 
    if w not in stop_words: 
       filtered_sentence.append(w) 

result=filtered_sentence	   
#print(result)
syn_result=[]
ant_result=[]


def similiarity(X_set,Y_set):
   l1 =[];l2 =[] 
   rvector = X_set.union(Y_set)# form a set containing keywords of both strings  
   for w in rvector: 
       if w in X_set: l1.append(1) # create a vector 
       else: l1.append(0) 
       if w in Y_set: l2.append(1) 
       else: l2.append(0) 
   c = 0
  
   # cosine formula  
   for i in range(len(rvector)): 
        c+= l1[i]*l2[i] 
   cosine = c / float((sum(l1)*sum(l2))**0.5) 
   return cosine

def jaccard_similarity(query, document):
    intersection = query.intersection(document)
    union = query.union(document)
    return len(intersection)/len(union)
	
def cosine_distance_wordembedding_method(s1, s2):
    import scipy
    vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
    vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')

def google_encoder_similiarity(sentences):
    import tensorflow as tf
    import tensorflow_hub as hub

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
    embed = hub.Module(module_url)
    #sentences = ["Python is a good language","Language a good python is"]

    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
    similarity_sentences_encodings = embed(similarity_input_placeholder)

    with tf.Session() as session:
       session.run(tf.global_variables_initializer())
       session.run(tf.tables_initializer())
       sentences_embeddings = session.run(similarity_sentences_encodings, feed_dict={similarity_input_placeholder: sentences})
       similarity = np.inner(sentences_embeddings[0], sentences_embeddings[1])
    print("Similarity is %s" % similarity)
  
for res in result:
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(res):
        for l in syn.lemmas():
           synonyms.append(l.name())
           if l.antonyms():
              antonyms.append(l.antonyms()[0].name())

    syn_result.append(synonyms)
    ant_result.append(antonyms)

#print(syn_result)
simil=[]
jaccard_simil=[]
for ind in train_text.index: 
    sentence=str(train_text['words'][ind])
    stop_words = set(stopwords.words('english')) 
    word_tokens = re.sub(r"[^a-zA-Z0-9]+", ' ', sentence).split(' ')
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words and len(w)>=3 :
           #print(w)		
           filtered_sentence.append(w) 
  
    #print(filtered_sentence)
    X_set = {w for w in filtered_sentence}
    Y_set = {w for w in result}
    if len(filtered_sentence)>=1:
       sim=similiarity(X_set,Y_set)
       simil.append(sim)
       jaccard_simil.append(jaccard_similarity(X_set,Y_set))
    else:
       simil.append(0)
       jaccard_simil.append(0)
    #str1=" ";str2=" "
    #QA=[str1.join(filtered_sentence),str2.join(result)]
    #print(QA)
    #google_encoder_similiarity(QA)	

#cosine similiarity of question with each sentence is found
#print(simil)

result_text= pd.DataFrame({'sentence':content_list,'cosine_similiarity':simil,'jaccard_similiarity':jaccard_simil})
#print(result_text)
result_text.to_csv('simils.csv') 

#for visualization purposes
result_text.plot(x='sentence', y='cosine_similiarity')
result_text.plot(x='sentence', y='jaccard_similiarity')
        	
max=result_text.max()			
max_cos=max.cosine_similiarity
max_jac=max.jaccard_similiarity

filter1 = result_text['cosine_similiarity']==max_cos
filter2 = result_text['jaccard_similiarity']==max_jac

res_record=result_text.loc[(result_text['cosine_similiarity'] == max_cos) & (result_text['jaccard_similiarity']==max_jac)]
res_sent=res_record.sentence.item()
print(res_sent)