from translate import Translator
translator = Translator(from_lang="autodetect",to_lang="en")
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import emoji

def traduci(text):
    return translator.translate(text)

def converti_emoji(text):
    return emoji.demojize(text)

def _get_lang_detector(nlp,name):
      return LanguageDetector()

nlp = spacy.load('en_core_web_sm')
Language.factory("language_detector", func=_get_lang_detector)
nlp.add_pipe('language_detector', last=True)

def get_lang_text(text):
    return nlp(text)._.language
