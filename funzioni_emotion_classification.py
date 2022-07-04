from sklearn.metrics import accuracy_score, classification_report
import funzioni_preprocessing_text as fpt
import joblib
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class ModelEmotionDetection:
  def __init__(self,xy_input=(None,None),save=False):
    if all(xy_input):
      bigramLS_post,self.bigram_mdl=fpt.clear_corpus(xy_input[0])
      counterBigram=[Counter(tmp) for tmp in bigramLS_post]
      #vectorizer = DictVectorizer(sparse = True)
      self.vectorizer = TfidfVectorizer()
      X_all=self.vectorizer.fit_transform(counterBigram)
      y_all=list(xy_input[1])
      X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)
      self.mnb = MultinomialNB()
      self.mnb.fit(X_train, y_train)
      self.asTrain,self.crTrain=self.get_information_model(y_train,X_train)
      self.asTest,self.crTest=self.get_information_model(y_test,X_test)
      if save:
        joblib.dump(self.mnb, 'modelEmotion/mnb.pkl')
        joblib.dump(self.vectorizer, 'modelEmotion/vectorizer.pkl')
        joblib.dump(self.bigram_mdl, 'modelEmotion/bigram_mdl.pkl')
        joblib.dump(self.asTrain, 'modelEmotion/asTrain.pkl')
        joblib.dump(self.asTest, 'modelEmotion/asTest.pkl')
        joblib.dump(self.crTrain, 'modelEmotion/crTrain.pkl')
        joblib.dump(self.crTest, 'modelEmotion/crTest.pkl')
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
    
  def get_prob(self,tweet):
      tmp = self.mnb.predict_proba(fpt.post_adapter(tweet,self.vectorizer,self.bigram_mdl))
      tmp2=list(tmp[0])
      class2=list(self.mnb.classes_)
      return sorted(dict(zip(class2,tmp2)).items(), key=lambda x: x[1], reverse=True)
  
  def get_emotion(self,tweet):
    return self.mnb.predict(fpt.post_adapter(tweet,self.vectorizer,self.bigram_mdl))[0]
  
  def get_information_model_train(self):
    print("Accuracy Train:",self.asTrain)
    print(self.crTrain)
  
  def get_information_model_test(self):
    print("Accuracy Test:",self.asTest)
    print(self.crTest)
  
  def get_information_model(self,y,X):
    return (accuracy_score(y, self.mnb.predict(X)),classification_report(y, self.mnb.predict(X),zero_division=0))