from sklearn.metrics import accuracy_score, classification_report
import funzioni_preprocessing_text as fpt
import joblib
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def get_information_model(model,y,X):
  return (accuracy_score(y, model.predict(X)),classification_report(y, model.predict(X),zero_division=0))

def create_model_and_get_components_emotion(emotion_file,save=False):
  df=pd.read_csv(emotion_file)
  df=df[["sentiment","content"]]
  bigramLS_post,bigram_mdl=fpt.clear_corpus(df["content"])
  df["content"]=[Counter(tmp) for tmp in bigramLS_post]
  #vectorizer = DictVectorizer(sparse = True)
  vectorizer = TfidfVectorizer()
  X_all=vectorizer.fit_transform(list(df["content"]))
  y_all=list(df["sentiment"])
  X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)
  mnb = MultinomialNB()
  mnb.fit(X_train, y_train)
  asTrain,crTrain=get_information_model(mnb,y_train,X_train)
  asTest,crTest=get_information_model(mnb,y_test,X_test)
  
  if save:
    joblib.dump(mnb, 'modelEmotion/mnb.pkl')
    joblib.dump(vectorizer, 'modelEmotion/vectorizer.pkl')
    joblib.dump(bigram_mdl, 'modelEmotion/bigram_mdl.pkl')
    joblib.dump(asTrain, 'modelEmotion/asTrain.pkl')
    joblib.dump(asTest, 'modelEmotion/asTest.pkl')
    joblib.dump(crTrain, 'modelEmotion/crTrain.pkl')
    joblib.dump(crTest, 'modelEmotion/crTest.pkl')

  return vectorizer,bigram_mdl,mnb,asTrain,crTrain,asTest,crTest

def load_model_and_get_components_emotion():
  try:
    vectorizer = joblib.load('modelEmotion/vectorizer.pkl')
    bigram_mdl = joblib.load('modelEmotion/bigram_mdl.pkl')
    mnb = joblib.load('modelEmotion/mnb.pkl')
    asTrain = joblib.load('modelEmotion/asTrain.pkl')
    asTest = joblib.load('modelEmotion/asTest.pkl')
    crTrain = joblib.load('modelEmotion/crTrain.pkl')
    crTest = joblib.load('modelEmotion/crTest.pkl')
    return vectorizer,bigram_mdl,mnb,asTrain,crTrain,asTest,crTest

  except:
    return None,None,None

################################
def list_prob(tweet,vectorizer,bigram_mdl,model):
    tmp = model.predict_proba(fpt.post_adapter(tweet,vectorizer,bigram_mdl))
    tmp2=list(tmp[0])
    class2=list(model.classes_)
    return sorted(dict(zip(class2,tmp2)).items(), key=lambda x: x[1], reverse=True)

def get_emotion(tweet,vectorizer,bigram_mdl,model):
    return model.predict(fpt.post_adapter(tweet,vectorizer,bigram_mdl))[0]

class ModelEmotionDetection:
  def __init__(self,emotion_file=None):
    self.vectorizer,self.bigram_mdl,self.model,\
    self.asTrain,self.crTrain,self.asTest,self.crTest=\
    create_model_and_get_components_emotion(emotion_file) if emotion_file else\
    load_model_and_get_components_emotion()
  
  def get_prob(self,tweet):
    return list_prob(tweet,self.vectorizer,self.bigram_mdl,self.model)
  
  def get_emotion(self,tweet):
    return get_emotion(tweet,self.vectorizer,self.bigram_mdl,self.model)
  
  def get_information_model_train(self):
    print("Accuracy Train:",self.asTrain)
    print(self.crTrain)
  
  def get_information_model_test(self):
    print("Accuracy Test:",self.asTest)
    print(self.crTest)