import google_trans_new as gtn
translator = gtn.google_translator()
import emoji

#Funzione che traduce un testo da qualsiasi lingua in inglese
def traduci(text):
    return text if get_lang_text(text)=="english" else translator.translate(text)

#Funzione che converte l'emoji in testo
def converti_emoji(text):
    return emoji.demojize(text)

#Funzione che ritorna la lingua di un testo
def get_lang_text(text):
    return translator.detect(text)[1]