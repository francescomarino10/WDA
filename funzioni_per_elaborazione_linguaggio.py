# from translate import Translator
# translator = Translator(from_lang="autodetect",to_lang="en")
import google_trans_new as gtn
translator = gtn.google_translator()
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import emoji

def traduci(text):
    return text if get_lang_text(text)=="english" else translator.translate(text)

def converti_emoji(text):
    return emoji.demojize(text)

# def _get_lang_detector(nlp,name):
#       return LanguageDetector()

# nlp = spacy.load('en_core_web_sm')
# Language.factory("language_detector", func=_get_lang_detector)
# nlp.add_pipe('language_detector', last=True)

# def get_lang_text(text):
#     return nlp(text)._.language

def get_lang_text(text):
    return translator.detect(text)[1]

