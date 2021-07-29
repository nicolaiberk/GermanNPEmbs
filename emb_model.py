## estimate word embeddings from newspaper data
## code adapted from https://github.com/damian0604/embeddingworkshop/blob/main/04exercise.ipynb
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import string
import re
import os
import pandas as pd
import csv


# tqdm allows you to display progress bars in loops
from tqdm import tqdm
from datetime import datetime

import gensim

# lets get more output
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# get full set of news articles
if not os.path.isfile('newspapers/_bild_articles.csv') and not os.path.isfile('uniquesentences.txt'):
    os.system('mkdir newspapers')
    os.system('wget -O newspapers/articles.zip https://www.dropbox.com/sh/r6k4qk9flgz0agu/AAA5ZLsuOwk9UWiEsLAOFmDSa?dl=0')
    os.system('unzip newspapers/articles.zip -d newspapers')
    os.system('rm newspapers/articles.zip')

              
if not os.path.isfile('uniquesentences.txt'):
    # load all texts
    for filename in tqdm(os.listdir('newspapers')):
      if 'artcls' in locals():
        print(f'\nLoaded {artcls.shape[0]} articles')
        artcls = artcls.append(pd.read_csv('newspapers/'+filename))
      else:
        artcls = pd.read_csv('newspapers/'+filename)
    print(f'Loaded {artcls.shape[0]} articles, done.')

    artcls = artcls.reset_index()


    # keep only if string
    stringvar = [str == type(i) for i in artcls.text]
    artcls = artcls[stringvar]

    # cut into sentences
    print('Cutting into sentences:')
    trans = str.maketrans('', '', string.punctuation) # translation scheme for removing punctuation
    uniquesentences = set()
    for review in tqdm(artcls.text):
        for sentence in sent_tokenize(review):
            # remove HTML tags in there
            sentence = re.sub(r"<.*?>"," ",sentence)
            sentence = sentence.translate(trans) 
            if sentence not in uniquesentences:
                uniquesentences.add(sentence.lower())

    print(f"We now have {len(uniquesentences)} unique sentences.")
              
    del(artcls)
    
    print('Saving uniquesentences.txt:')
    with open('uniquesentences.txt', mode='w') as fo:
      for sentence in tqdm(uniquesentences):
        fo.write(sentence)

    del(uniquesentences)
    os.system('rm newspapers -r')


# loading unique sentences
print('Loading unique sentences...')
tokenizedsentences = []
with open('drive/MyDrive/uniquesentences.txt', mode='r') as fi:
  reader = csv.reader(fi)
  for sentence in tqdm(reader):
    tokenizedsentences.append(sentence.split())
    
print(f"Started setting up the model at {datetime.now()}")
model = gensim.models.Word2Vec(size=300, min_count=100) # we want 300 dimensions and not overdo it with the features
model.build_vocab(tokenizedsentences)
print(f"Started training at {datetime.now()}")
model.train(tokenizedsentences, total_examples=model.corpus_count,  epochs=1)
print(f"Finished training at {datetime.now()}")

print('Saving model:')
model.save("np_emb")
print('Model finished!')
