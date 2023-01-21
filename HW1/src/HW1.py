#Python Version: 3.7.9
#Pandas Version: 1.3.5
#Numppy Version: 1.21.6
#NLTK Version: 3.8.1

import pandas as pd
import numpy as np
import nltk as nltk
import re
import contractions as ct
from textblob import TextBlob

import pkg_resources
from symspellpy import SymSpell

import warnings


from nltk.corpus import wordnet
from bs4 import BeautifulSoup

RANDOM_SAMPLE_SIZE = 20000

def data_cleaning(df):

    for i in range(0, 50):
        review_text = class1_df['review_body'][i]

        if BeautifulSoup(review_text, "html.parser").find():
            review_text = BeautifulSoup(review_text, "html.parser").get_text("　")

        review_text = review_text.lower()
        review_text = ct.fix(review_text)


        regex = re.compile('[^a-zA-Z]')
        review_text = regex.sub(' ', review_text)

        df.loc[i, ['review_body']] = review_text
        print(class1_df['review_body'][i] + "\n")

    return df

def init_data(df):

  df.dropna(inplace=True)
  df.drop_duplicates(inplace=True, subset=['review_body'])
  df['star_rating'] = df['star_rating'].astype('int')
  return df


if __name__ == '__main__':
    """
  warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
  df = pd.read_pickle("./data.pkl")

  df = init_data(df).reset_index(drop=True)

  class1_df = df[df['star_rating'] <= 2].reset_index(drop=True) #.sample(RANDOM_SAMPLE_SIZE)
  class2_df = df[df['star_rating'] == 3].sample(RANDOM_SAMPLE_SIZE)
  class3_df = df[df['star_rating'] >= 4].sample(RANDOM_SAMPLE_SIZE)

  class1_df = data_cleaning(class1_df)
"""

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

input_term = (
    "whereis th elove hehad dated forImuch of thepast who "
    "couqdn'tread in sixtgrade and ins pired him"
)
suggestions = sym_spell.lookup_compound(input_term, max_edit_distance=2)
print(suggestions[0].term)





