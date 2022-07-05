"""Funzioni per la pulizia del testo"""

import re
import gensim
from nltk.stem import WordNetLemmatizer
from numpy import spacing
import funzioni_per_elaborazione_linguaggio as fel
from collections import Counter
import spacy
ENGLISH_STOP_WORDS=spacy.load('en_core_web_sm').Defaults.stop_words

def remove_link_menzione(doc):
    pattern_link="https\S+"
    pattern_menzione="@\S+"
    doc=re.sub(pattern_link, '',doc)
    doc=re.sub(pattern_menzione," ",doc)
    return doc

def find_and_merge_not(lista):#unisce not parola --> not_parola
    tmp=''
    parole_esaminate=[]
    for parole in lista:
        if parole == 'not':
            tmp='not_'
        elif tmp=='not_' and parole not in ENGLISH_STOP_WORDS:
            parole=tmp+parole
            tmp=''
        if parole != 'not' and parole not in ENGLISH_STOP_WORDS:
            parole_esaminate.append(parole)
    return parole_esaminate

def clear_corpus(tweets):#return bigram
    allowed_postags=["NOUN", "ADJ", "VERB", "ADV","PROPN","PART"]
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    lemmatized_string = [\
        find_and_merge_not(\
            gensim.utils.simple_preprocess(\
                remove_link_menzione(\
                    " ".join([token.lemma_ for token in nlp(tweet) if token.pos_ in allowed_postags])), \
                        deacc=True))\
                         for tweet in tweets]

    bigram_phrases = gensim.models.phrases.Phrases(lemmatized_string, min_count=1, threshold=2)
    bigram_mdl = gensim.models.phrases.Phraser(bigram_phrases)

    post=[bigram_mdl[l] for l in lemmatized_string]

    return post,bigram_mdl

def post_adapter(post,bigram_mdl,vectorizer=None):
    allowed_postags=["NOUN", "ADJ", "VERB", "ADV","PROPN","PART"]
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    lemmatized_string=find_and_merge_not(\
        gensim.utils.simple_preprocess(\
            remove_link_menzione(\
                " ".join([token.lemma_ for token in nlp(fel.converti_emoji(fel.traduci(post))) if token.pos_ in allowed_postags])), \
                        deacc=True))

    post_clear=bigram_mdl[lemmatized_string]
    return vectorizer.transform([Counter(post_clear)]) if vectorizer else post_clear