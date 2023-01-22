# Python Version: 3.7.9
# Pandas Version: 1.3.5
# Numppy Version: 1.21.6
# NLTK Version: 3.8.1

import pandas as pd
import numpy as np
import nltk as nltk
import re

from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup

import contractions as ct
import pkg_resources
from symspellpy import SymSpell
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import map_tag, WordNetLemmatizer, pos_tag
import warnings

RANDOM_SAMPLE_SIZE = 20000


def init_spell_checker():
    sym_spell_obj = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
    )
    sym_spell_obj.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell_obj.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    return sym_spell_obj


def spell_correct(text):
    input_term = text
    suggestions = sym_spell.lookup_compound(
        input_term, max_edit_distance=2, transfer_casing=True
    )
    return suggestions[0].term


def remove_stopwords(review_body_string):
    word_tokens = word_tokenize(review_body_string)
    buffer_string = ""

    for w in word_tokens:
        if w not in stop_words:
            w = word_lemmatization(w)
            buffer_string = buffer_string + w + "　"

    buffer_string = re.sub(' +', ' ', buffer_string).strip()

    return buffer_string


def get_adverb_lemma(word):
    temp_bool = False
    param_wn_synset = word + ".r.01"

    # check if word's synset continas adverb option
    for i in range(0, len(wn.synsets(word))):
        if param_wn_synset == str((wn.synsets(word)[i])).split("\'")[1]:
            temp_bool = True

    if  temp_bool == False:
        return word

    # if yes and suggestion not empty then return suggestion, else return original word
    suggest_lemma_list = wn.synset(param_wn_synset).lemmas()[0].pertainyms()
    if len(suggest_lemma_list) > 0:
        return suggest_lemma_list[0].name()
    else:
        return word

def word_lemmatization(word):
    treebank_pos_tag = pos_tag([word])[0][1]
    universal_pos_tag = map_tag('en-ptb', 'universal', treebank_pos_tag)

    if universal_pos_tag == "ADJ":
        word = wnl.lemmatize(word, wn.ADJ)
    elif universal_pos_tag == "VERB":
        word = wnl.lemmatize(word, wn.VERB)
    elif universal_pos_tag == "NOUN":
        word = wnl.lemmatize(word, wn.NOUN)
    elif universal_pos_tag == "ADV":
        word = wnl.lemmatize(word, wn.ADV)
        word = get_adverb_lemma(word)

    return word


def data_cleaning(data_frame):
    # 0-50 for testing purpose
    for i in range(0, 50):
        review_text = class1_df['review_body'][i]

        # remove un-wanted html tags
        if BeautifulSoup(review_text, "html.parser").find():
            review_text = BeautifulSoup(review_text, "html.parser").get_text("　")

        # spell correction
        review_text = spell_correct(review_text)

        # text extend contractions
        review_text = ct.fix(review_text)

        # remove non-alphabetical chars
        regex = re.compile('[^a-zA-Z]')
        review_text = regex.sub(' ', review_text)

        # lower case and strip
        review_text = review_text.lower().strip()

        print(review_text)
        # remove stop words
        review_text = remove_stopwords(review_text)

        data_frame.loc[i, ['review_body']] = review_text
        print(class1_df['review_body'][i] + "\n")

    return data_frame


def init_data(data_frame):
    data_frame.dropna(inplace=True)
    data_frame.drop_duplicates(inplace=True)
    data_frame['star_rating'] = data_frame['star_rating'].astype('int')
    return data_frame


if __name__ == '__main__':

    # init
    warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
    df = pd.read_pickle("./data.pkl")
    sym_spell = init_spell_checker()
    df = init_data(df).reset_index(drop=True)
    stop_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    # 3-classes dataset
    class1_df = df[df['star_rating'] <= 2].sample(RANDOM_SAMPLE_SIZE).reset_index(drop=True)
    class2_df = df[df['star_rating'] == 3].sample(RANDOM_SAMPLE_SIZE)
    class3_df = df[df['star_rating'] >= 4].sample(RANDOM_SAMPLE_SIZE)

    # working on class1_df to test code, handle duplicated rows that review_body only differ in whitespaces
    class1_df = data_cleaning(class1_df)
    print(class1_df.info(verbose="True"))

    class1_df.drop_duplicates(inplace=True)
    print(class1_df.info(verbose = "True"))

    class1_df.drop_duplicates(inplace=True, subset="review_body")
    print(class1_df.info(verbose="True"))








