import re
import gensim
import funzioni_per_elaborazione_linguaggio as fel
import spacy
ENGLISH_STOP_WORDS=spacy.load('en_core_web_sm').Defaults.stop_words

#Funzione per rimuovere link e menzioni dal testo
def remove_link_menzione(doc):
    pattern_link="https\S+"
    pattern_menzione="@\S+"
    doc=re.sub(pattern_link, '',doc)
    doc=re.sub(pattern_menzione," ",doc)
    return doc

#Funzione che unisce not con una parola che segue
#Esempio "not","good" -> "not_good"
def find_and_merge_not(lista):
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

#Funzione che pulisce il corpus di una lista di post presi in input
#considerando le categorie di parole più utili per i nostri scopi
#eliminando stop word, tokenizzando il testo, creando bigrammi
#Utilizzato per l'addestramento dei modelli
#dove non era necessaria la conversione delle emoji perchè non aggiungeva 
# informazione utile per la rilevazione del topic e poichè nel dataset delle emotion non erano presenti
#e la traduzione perchè ogni post era in inglese
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

#Funzione per la pulizia di un post in input
#Come clear_corpus
#In aggiunta abbiamo la conversione delle emoji e la traduzione
#Se passato un vettorizzatore il posto viene vettorizzato
#Utilizzato per i commenti di un post dove ogni emoji può dare informazione in più
#e dove è necessaria una traduzione
def post_adapter(post,bigram_mdl,vectorizer=None,traduci=False):
    allowed_postags=["NOUN", "ADJ", "VERB", "ADV","PROPN","PART"]
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    lemmatized_string=find_and_merge_not(\
        gensim.utils.simple_preprocess(\
            remove_link_menzione(\
                " ".join([token.lemma_ for token in nlp(fel.converti_emoji(fel.traduci(post) if traduci else post)) if token.pos_ in allowed_postags])), \
                        deacc=True))

    post_clear=bigram_mdl[lemmatized_string]
    lemmatized_bigram_string=' '.join(post_clear)
    return vectorizer.transform([lemmatized_bigram_string]) if vectorizer else post_clear