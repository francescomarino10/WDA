
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
import spacy
import gensim
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

#funzioni per la pulizia del corpus
def remove_link_menzione(doc):
    pattern_link="https\S+"
    pattern_menzione="@\S+"
    doc=re.sub(pattern_link, '',doc)
    doc=re.sub(pattern_menzione," ",doc)
    return doc

def find_and_merge_not(lista,all_stopwords):#unisce not parola --> not_parola
    tmp=''
    parole_esaminate=[]
    for parole in lista:
        if parole == 'not':
            tmp='not_'
        elif tmp=='not_' and parole not in all_stopwords:
            parole=tmp+parole
            tmp=''
        if parole != 'not' and parole not in all_stopwords:
            parole_esaminate.append(parole)
    return parole_esaminate

def clear_corpus(tweets):#return bigram                
    tokens_ = [list(gensim.utils.tokenize(remove_link_menzione(tweet), lower=True)) for tweet in tweets]
    
    sp = spacy.load('en_core_web_sm')
    all_stopwords = sp.Defaults.stop_words

    wnl = WordNetLemmatizer()
    lemmatized_string=[[wnl.lemmatize(words) for words in find_and_merge_not(lista,all_stopwords)] for lista in tokens_]
    bigram_mdl = gensim.models.phrases.Phrases(lemmatized_string, min_count=1, threshold=2)##farglielo fare su tutto il corpus tokenizzato altrimenti non funziona se lo fai su una sola frase
    
    post=[bigram_mdl[l] for l in lemmatized_string]
    
    #lemmatized_bigram_string=[' '.join(word) for word in post]
    return post,bigram_mdl

#funzioni topic recognition
def tf_idfLSA(lemmatized_bigram_string):
  vectorizer = TfidfVectorizer()
  M=vectorizer.fit_transform(lemmatized_bigram_string)
  termini=vectorizer.get_feature_names()
  return vectorizer,M,termini

def lsa(M,termini):
  lsa = TruncatedSVD(n_components=10,n_iter=300)
  lsa.fit(M)
  return lsa

def get_concettiLSA(lsa,termini):
  concetti={}
#usare questo quelli di prima sono prove
  for i,j in enumerate(lsa.components_):
    parolaChiave=zip(termini,abs(j))
    paroleOrdinate = sorted(parolaChiave,key=lambda x:x[1], reverse = True)
    paroleOrdinate = paroleOrdinate[:10]
    concetti["concetto_" + str(i)]=[]
    #print("Asse concettuale n. ",i, ": ")
    for k in paroleOrdinate:
      concetti["concetto_" + str(i)].append(k)
      #print(k)
    #print('\n\n')
  return concetti

def punteggi_LSA(concetti,parole):
  punteggi=dict()

  for key,lista in concetti.items():
    punteggio=0
    for i in parole:
      for concetto_parola,valore in lista:
        if(i == concetto_parola):
          punteggio+=valore
    punteggi[key]=punteggio
  return punteggi

def result_TopicLSA(punteggi,concetti):
  risultato = []
  for concetto,valore in punteggi.items():
    if valore!=0:
      print(concetto, concetti[concetto])
  return risultato

def result_Topic_max(punteggi,concetti):
  risultato = []
  massimo = max(punteggi.values())
  for concetto,valore in punteggi.items():
    if valore == massimo:
      risultato.append((concetto, concetti[concetto]))
  return risultato

def create_lsa_model(corpus):
  post,bigram_mdl=clear_corpus(corpus)
  lemmatized_bigram_string=[' '.join(word) for word in post]
  vectorizer,M,termini=tf_idfLSA(lemmatized_bigram_string)
  lsa=lsa(M,termini)
  termini=vectorizer.get_feature_names()
  concetti=get_concettiLSA(lsa)
  return bigram_mdl,vectorizer,concetti

#trova
def post_adapter_lsa(post,vectorizer,bigram_mdl):
    
    sp = spacy.load('en_core_web_sm')
    all_stopwords = sp.Defaults.stop_words

    wnl = WordNetLemmatizer()


def lda(post):
  parole_valore=[]
  concetti_LDA={}

  dictionary = gensim.corpora.Dictionary(post)
  corpus2 = [dictionary.doc2bow(text) for text in post]

  tfidf = gensim.models.TfidfModel(corpus2)
  corpus_tfidf = tfidf[corpus2]

  ldamodel = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=10, id2word=dictionary, passes=500)
  

  for idx,lista_t in topics:
    parole_valore=[]
    tmp_p=re.sub('[\"]',"",lista_t)
    tmp_p=tmp_p.split(' + ') 
    coppia=list(t.split('*') for t in tmp_p)
    concetti_LDA['concetto_'+str(idx)]=[]
    for valore,parola in coppia:
      concetti_LDA['concetto_'+str(idx)].append((parola,valore))
  return ldamodel,concetti_LDA,corpus2

def punteggioLDA(concetti_LDA,parole):
  punteggi=dict()
  for key in concetti_LDA.keys():
    punteggio=0
    for i in parole:
      for concetto_parola,valore in concetti_LDA[key]:
        if(i == concetto_parola):
         punteggio+=float(valore)
    punteggi[key]=punteggio
  return punteggi

def result_TopicLDA(concetti_LDA,parole):
  risultato = []
  p=punteggioLDA(concetti_LDA,parole)
  massimo = max(p.values())
  for concetto,valore in p.items():
    if valore == massimo:
      risultato.append((concetto, concetti_LDA[concetto]))
  return (risultato)

def grafico_LDA(ldamodel,corpus2,dictionary):

  pyLDAvis.enable_notebook()
  lda_display = gensimvis.prepare(ldamodel, corpus2, dictionary, sort_topics=False)

  pyLDAvis.display(lda_display)

def hdp(corpus2):
  hdpmodel = gensim.models.HdpModel(corpus=corpus2, id2word=dictionary)
  hdptopics = hdpmodel.show_topics(formatted=False)
  return hdpmodel,hdptopics

def punteggioHDP(hdptopics,parole):
  punteggi=dict()
  risultato = []
  for key,lista in hdptopics:
    punteggio=0
    for i in parole:
      for concetto_parola,valore in lista:
        if(i == concetto_parola):
         punteggio+=valore
    if punteggio!=0:
        risultato.append((key,lista))
    punteggi[key]=punteggio
  return punteggi

def result_TopicHDP(hdptopics,parole):
  risultato = []
  p=punteggioHDP(hdptopics,parole)
  massimo = max(p.values())
  #print("Risultato di ", corpus)
  for concetto,valore in p.items():
    if valore == massimo:
      risultato.append((concetto, hdptopics[concetto]))
  return (risultato)

"""###########################funzioni emotion detection########################"""
##inizio emotion##
def get_information_model_train(model,y_train,X_train):
  print("Accuracy Train",accuracy_score(y_train, model.predict(X_train)))
  print(classification_report(y_train, model.predict(X_train),zero_division=0))

def get_information_model_test(model,y_test,X_test):
  print("Accuracy Test",accuracy_score(y_test, model.predict(X_test)))
  print(classification_report(y_test, model.predict(X_test),zero_division=0))

def create_model_and_get_components_emotion(emotion_file,print_inf=False,save=False):
  df=pd.read_csv(emotion_file)
  df=df[["sentiment","content"]]
  bigramLS_post,bigram_mdl=clear_corpus(df["content"])
  df["content"]=[Counter(tmp) for tmp in bigramLS_post]
  vectorizer = DictVectorizer(sparse = True)
  X_all=vectorizer.fit_transform(list(df["content"]))
  y_all=list(df["sentiment"])
  X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)
  mnb = MultinomialNB()
  mnb.fit(X_train, y_train)
  if print_inf:
    get_information_model_train(mnb,y_train,X_train)
    get_information_model_test(mnb,y_test,X_test)
  
  if save:
    joblib.dump(mnb, 'mnb.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(bigram_mdl, 'bigram_mdl.pkl')

  return vectorizer,bigram_mdl,mnb

def load_model_and_get_components_emotion():
  try:
    vectorizer = joblib.load('vectorizer.pkl')
    bigram_mdl = joblib.load('bigram_mdl.pkl')
    mnb = joblib.load('mnb.pkl')
    return vectorizer,bigram_mdl,mnb
  except:
    return None,None,None

def tweet_adapter(tweet,vectorizer,bigram_mdl):
    lista_token = gensim.utils.tokenize(remove_link_menzione(tweet), lower=True)
    
    sp = spacy.load('en_core_web_sm')
    all_stopwords = sp.Defaults.stop_words

    wnl = WordNetLemmatizer()
    lemmatized_string=[wnl.lemmatize(words) for words in find_and_merge_not(lista_token,all_stopwords)]
        
    post=bigram_mdl[lemmatized_string]
    return vectorizer.transform([Counter(post)])

def list_prob(tweet,vectorizer,bigram_mdl,model):
    tmp = model.predict_proba(tweet_adapter(tweet,vectorizer,bigram_mdl))
    tmp2=list(tmp[0])
    class2=list(model.classes_)
    return sorted(dict(zip(class2,tmp2)).items(), key=lambda x: x[1], reverse=True)

def get_emotion(tweet,vectorizer,bigram_mdl,model):
    return model.predict(tweet_adapter(tweet,vectorizer,bigram_mdl))[0]

class ModelEmotionDetection:
  def __init__(self,emotion_file=None):
    if emotion_file is None:
      self.vectorizer,self.bigram_mdl,self.model=load_model_and_get_components_emotion()
    else:
      self.vectorizer,self.bigram_mdl,self.model=create_model_and_get_components_emotion(emotion_file)
  
  def get_prob(self,tweet):
    return list_prob(tweet,self.vectorizer,self.bigram_mdl,self.model)
  
  def get_emotion(self,tweet):
    return get_emotion(tweet,self.vectorizer,self.bigram_mdl,self.model)
  
##fine emotion##