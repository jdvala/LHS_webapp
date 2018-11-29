#######################################################################
# English Preprocessing Script
# Created on: 18/05/2018
# According to the post dated on: May 4, 2018
# Author: Jay Vala
#######################################################################

import os
import sys
import string
import re
import spacy
from nltk.corpus import stopwords

# Loading spacy english model for lemmetization

nlp = spacy.load('en')



# Remove text with regular expressions

def remove(text):
    """Returns text with all the filtering necessary
    :params: Text as sentence as type string
    :returns: manipulated sentence as type string"""

    t = re.sub(r"(\d+\.\d+)","",text)
    #t = re.sub(r"(\d+th?|st?|nd?|rd?)","", t)
    t = re.sub(r"\d{2}.\d{2}.\d{4}","",t)
    t = re.sub(r"\d{2}\/\d{2}\/\d{4}","",t)
    t = re.sub(r"\d{2}(\/|\.)\d{2}(\/|\.)\d{2}","",t)
    t = re.sub(r"($|€|¥|₹|£)","",t)
    t = re.sub(r"(%)","",t)
    t = re.sub(r"\d+","",t)
    t = re.sub(r"\n","",t)
    t = re.sub(r"\xa0", "", t)
    return t


# Remove puntuations

def pun(text):
    """Return punctuations from text
    :params: Text as sentence as type string
    :returns: manipulated sentence as type string"""

    table = str.maketrans("","", string.punctuation)
    t = text.translate(table)
    return t

# Lemmetizer

def lemmatizer(text):
    """Returns text after lemmatization
    :params: Text as sentence as type string
    :returns: manipulated sentence(lemmetized) as type string
    """
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)


def extras(text):
    """Returns text after removing some extra symbols
     :params: Text as sentence as type string
    :returns: manipulated sentencs as type string"""

    t = re.sub(r"\"|\—|\'|\’\•","",text)
    word_list = t.split()
    for index, word in enumerate(word_list):
        if len(word) <=1:
            del word_list[index]
    t = ' '.join(word_list)

    return t


# Stop word removal
def stop_word(text):
    """Returns text after removing english stop words
    :params: Text as sentence as type string
    :returns: manipulated sentencs as type string"""

    list_ = []
    stop_words = stopwords.words('english')
    words_list = text.split()
    for word in words_list:
        if word not in stop_words:
            list_.append(word)
    return ' '.join(list_)


def main(sentence):
    """Main Function
    :params: sentence as text
    :returns: preprocessed text as string"""

    print("Starting to preprocess...")
    # Removing stop words
    t1 = stop_word(sentence)
    # lemmatization
    t2 = lemmatizer(t1)
    # Removing all the unnecessary things from the text
    t3 = remove(t2)
    # Removing punctuations
    t4 =pun(t3.lower())
    # Removing extras
    t5 = extras(t4)

    words = [word for word in t5.split() if len(word)>2]

    print('Preprocessing done.')
    
    return ' '.join(words)

if __name__ == "__main__":
    main()
