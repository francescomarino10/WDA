from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim
import funzioni_preprocessing_text as fpt
import itertools
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

#-----------LSA----------#
class LSAModel:
  """
  Parameters
  ----------
  corpus : list of str
  num_topics : number of topics to be extracted
  """
  def __init__(self,corpus,num_topics=10):
    post,self.bigram_mdl=fpt.clear_corpus(corpus)
    lemmatized_bigram_string=[' '.join(word) for word in post]
    self.vectorizer = TfidfVectorizer()
    M=self.vectorizer.fit_transform(lemmatized_bigram_string)
    self.termini=self.vectorizer.get_feature_names()
    self.lsa = TruncatedSVD(n_components=num_topics,n_iter=300)
    self.lsa.fit(M)
    self.concetti=self.print_topics()
  
  def print_topics(self,num_words=5):
    return {
        f"concetto_{str(i)}": list(
            sorted(zip(self.termini, abs(j)), key=lambda x: x[1],
                   reverse=True)[:num_words])
        for i, j in enumerate(self.lsa.components_)
    }

  def punteggi_LSA(self,parole):
    return {key : sum(valore for i, (concetto_parola,
                                valore) in itertools.product(parole, lista)
                if i == concetto_parola)
        for key, lista in self.concetti.items()
    }

  def result_TopicLSA(self,post):
    post_clear = fpt.post_adapter(post,self.bigram_mdl)
    punteggi_LSA=self.punteggi_LSA(post_clear)
    return [(concetto, self.concetti[concetto])
          for concetto, valore in punteggi_LSA.items() if valore != 0]
  
  def result_TopicLSAMax(self,post):
    post_clear = fpt.post_adapter(post,self.bigram_mdl)
    punteggi_LSA=self.punteggi_LSA(post_clear)
    massimo = max(punteggi_LSA.values())
    return [(concetto, self.concetti[concetto])
            for concetto, valore in punteggi_LSA.items() if valore == massimo]

#-----------LDA----------#
class LDAModel:
  """
  Parameters
  ----------
  corpus : list of str
  num_topics : number of topics to be extracted
  """
  def __init__(self,corpus,num_topics=10):
    post,self.bigram_mdl=fpt.clear_corpus(corpus)
    self.dictionary = gensim.corpora.Dictionary(post)
    corpus2 = [self.dictionary.doc2bow(text) for text in post]
    tfidf = gensim.models.TfidfModel(corpus2)
    self.corpus_tfidf = tfidf[corpus2]
    self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus_tfidf, num_topics, id2word=self.dictionary)
  
  def result_TopicLDA(self,post):
    post2 = fpt.post_adapter(post, self.bigram_mdl)
    post3 = self.dictionary.doc2bow(post2)
    return self.ldamodel.get_document_topics(post3)

  def result_TopicLDAMax(self,post):
    result = self.result_TopicLDA(post)
    massimo = max(result, key=lambda x: x[1])
    return self.ldamodel.print_topic(massimo, topn=5)
  
  def print_topics(self,num_words=5):
    return self.ldamodel.print_topics(num_words=num_words)
  
  def grafico_LDA(self):
    pyLDAvis.enable_notebook()
    lda_display = gensimvis.prepare(self.ldamodel, self.corpus_tfidf, self.dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)

#-----------HDP----------#
class HDPModel:
  """
  Parameters
  ----------
  corpus : list of str
  """
  def __init__(self,corpus):
    post,self.bigram_mdl=fpt.clear_corpus(corpus)
    self.dictionary = gensim.corpora.Dictionary(post)
    corpus2 = [self.dictionary.doc2bow(text) for text in post]
    tfidf = gensim.models.TfidfModel(corpus2)
    corpus_tfidf = tfidf[corpus2]
    self.hdpmodel = gensim.models.HdpModel(corpus_tfidf, self.dictionary)

  def result_TopicHDP(self,post):
    post2 = fpt.post_adapter(post, self.bigram_mdl)
    post3 = self.dictionary.doc2bow(post2)
    return self.hdpmodel.get_document_topics(post3)
  
  def result_TopicHDPMax(self,post):
    result = self.result_TopicHDP(post)
    massimo = max(result, key=lambda x: x[1])
    return self.hdpmodel.print_topic(massimo, topn=5)
  
  def print_topics(self,num_words=5):
    return self.hdpmodel.print_topics(num_words=num_words)