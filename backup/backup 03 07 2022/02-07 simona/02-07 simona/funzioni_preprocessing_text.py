"""Funzioni per la pulizia del testo"""

import re
import gensim
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
import funzioni_per_elaborazione_linguaggio as fel

def remove_link_menzione(doc):
    pattern_link="https\S+"
    pattern_menzione="@\S+"
    doc=re.sub(pattern_link, '',doc)
    doc=re.sub(pattern_menzione," ",doc)
    return doc

def find_and_merge_not(lista,ENGLISH_STOP_WORDS):#unisce not parola --> not_parola
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
    tokens_ = [list(gensim.utils.tokenize(remove_link_menzione(tweet), lower=True)) for tweet in tweets]
    
    #sp = spacy.load('en_core_web_sm')
    #all_stopwords = sp.Defaults.stop_words

    wnl = WordNetLemmatizer()
    lemmatized_string=[[wnl.lemmatize(words) for words in find_and_merge_not(lista,ENGLISH_STOP_WORDS)] for lista in tokens_]
    bigram_mdl = gensim.models.phrases.Phrases(lemmatized_string, min_count=1, threshold=2)##farglielo fare su tutto il corpus tokenizzato altrimenti non funziona se lo fai su una sola frase
    
    post=[bigram_mdl[l] for l in lemmatized_string]
    
    #lemmatized_bigram_string=[' '.join(word) for word in post]
    return post,bigram_mdl

def post_adapter(post,bigram_mdl,vectorizer=None):
    #sp = spacy.load('en_core_web_sm')
    #all_stopwords = sp.Defaults.stop_words
    wnl = WordNetLemmatizer()
    lista_token = list(gensim.utils.tokenize(remove_link_menzione(fel.converti_emoji(fel.traduci(post))), lower=True))
    lemmatized_string=[wnl.lemmatize(words) for words in find_and_merge_not(lista_token,ENGLISH_STOP_WORDS)]
    post_clear = bigram_mdl[lemmatized_string]
    return vectorizer.transform(post_clear) if vectorizer else post_clear