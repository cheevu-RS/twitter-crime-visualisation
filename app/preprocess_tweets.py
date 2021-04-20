import preprocessor as p
import pandas as pd
import numpy as np
import re
#important libraries for preprocessing using NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer

def remove_digits_lower_case(df):
    df = df.astype(str).str.replace('\d+', '')
    lower_text = df.str.lower()
    return lower_text

def remove_punctuation(words):
 print(words)
 new_words = []
 for word in words:
    new_word = re.sub(r'[^\w\s]', '', (word))
    if new_word != '':
       new_words.append(new_word)
 return new_words

lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = TweetTokenizer()

def lemmatize_text(text):
 return [(lemmatizer.lemmatize(w)) for w in \
                                     w_tokenizer.tokenize((text))]

stop_words = set(stopwords.words('english'))
stop_words.update("amp","","th")

TRAIN_DATASET_PATH = "../datasets/250tweets-dataset.csv"
tweets = pd.read_csv(TRAIN_DATASET_PATH)

for i,v in enumerate(tweets['tweet']):
    tweets.loc[i,'text'] = p.clean(v)

tweets['text'] = remove_digits_lower_case(tweets['text'])

tweets["text"] = tweets['text'].str.replace('[^\w\s]',' ')

tweets['text'] = tweets.text.apply(lemmatize_text)

tweets['text'] = tweets['text'].apply(lambda x: ' '.join([item for item in x if item not in stop_words]))

# for i in range(len(tweets)):
#     print(tweets['tweet'][i])
#     print(tweets['text'][i])
    
samples = tweets['text'].to_list()
labels = tweets['category'].to_list()
class_names = ["AntiSocialBehaviour","BikeTheft","Burglary","CriminalDamage","DrugOffences","PossessionOfWeapons","PublicOrder","Robbery","Shoplifting","TheftFromThePerson","VehicleCrime","ViolentCrime","Other","Terrorism"]