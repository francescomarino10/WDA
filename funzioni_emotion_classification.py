from sklearn.metrics import accuracy_score, classification_report
import funzioni_preprocessing_text as fpt
import joblib
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#Modello di Emotion Detection
class ModelEmotionDetection:
  def __init__(self,xy_input=(None,None),save=False):
    #Se passo i dati in input mi creo il modello
    if xy_input[0] is not None and xy_input[1] is not None:
      #Passo le frasi alla funzione clear_corpus, che mi restituisce i bigrammi e il modello bigram
      bigramLS_post,self.bigram_mdl=fpt.clear_corpus(xy_input[0])
      #Per ogni post ho un array di parola, le unisco in una stringa, affichè possa essere passato al fit_transform
      lemmatized_bigram_string=[' '.join(word) for word in bigramLS_post]
      #Vettorizziamo i post ottenedo una matrice sparsa in cui ogni cella contiene il valore TF-IDF delle coppie parola-post
      self.vectorizer = TfidfVectorizer()
      X_all=self.vectorizer.fit_transform(lemmatized_bigram_string)
      #Etichette di emotion per il training
      y_all=list(xy_input[1])
      #Splittiamo il training in training e test con una percentuale di test pari a 20%
      X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)
      #Inizializzo il modello MultinomialNB
      #Abbiamo scelto di utilizzare MultinomialNB perchè adatto al TF-IDF
      self.mnb = MultinomialNB()
      #Fit del modello
      self.mnb.fit(X_train, y_train)
      #Calcolo dell'accuracy score e del classification report del training
      self.asTrain,self.crTrain=self.get_information_model(y_train,X_train)
      #Calcolo dell'accuracy score e del classification report del test
      self.asTest,self.crTest=self.get_information_model(y_test,X_test)
      #Salvataggio del modello
      if save:
        joblib.dump(self.mnb, 'modelEmotion/mnb.pkl')
        joblib.dump(self.vectorizer, 'modelEmotion/vectorizer.pkl')
        joblib.dump(self.bigram_mdl, 'modelEmotion/bigram_mdl.pkl')
        joblib.dump(self.asTrain, 'modelEmotion/asTrain.pkl')
        joblib.dump(self.asTest, 'modelEmotion/asTest.pkl')
        joblib.dump(self.crTrain, 'modelEmotion/crTrain.pkl')
        joblib.dump(self.crTest, 'modelEmotion/crTest.pkl')
    #Se non passo i dati in input mi carico il modello
    else:
      try:
        self.vectorizer = joblib.load('modelEmotion/vectorizer.pkl')
        self.bigram_mdl = joblib.load('modelEmotion/bigram_mdl.pkl')
        self.mnb = joblib.load('modelEmotion/mnb.pkl')
        self.asTrain = joblib.load('modelEmotion/asTrain.pkl')
        self.asTest = joblib.load('modelEmotion/asTest.pkl')
        self.crTrain = joblib.load('modelEmotion/crTrain.pkl')
        self.crTest = joblib.load('modelEmotion/crTest.pkl')
      except:
        print("Errore caricamento modello")
  
  #Funzione che ritorna una lista di emotion con le relative probabilità ordinate
  def get_prob(self,tweet):
      tmp = self.mnb.predict_proba(fpt.post_adapter(tweet,self.bigram_mdl,self.vectorizer))
      tmp2=list(tmp[0])
      class2=list(self.mnb.classes_)
      return sorted(dict(zip(class2,tmp2)).items(), key=lambda x: x[1], reverse=True)
  
  #Funzione che ritorna l'emozione più probabile di un testo
  def get_emotion(self,tweet,translate=False):
    return self.mnb.predict(fpt.post_adapter(tweet,self.bigram_mdl,self.vectorizer,traduci=translate))[0]
  
  #Funzione che stampa l'accuracy score e il classification report del training
  def get_information_model_train(self):
    print("Accuracy Train:",self.asTrain)
    print(self.crTrain)
  
  #Funzione che stampa l'accuracy score e il classification report del test
  def get_information_model_test(self):
    print("Accuracy Test:",self.asTest)
    print(self.crTest)
  
  #Funzione che ritorna l'accuracy score e il classification report di dati in input
  def get_information_model(self,y,X):
    return (accuracy_score(y, self.mnb.predict(X)),classification_report(y, self.mnb.predict(X),zero_division=0))