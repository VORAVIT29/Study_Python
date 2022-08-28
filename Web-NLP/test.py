import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(u"UN,ICS,Greece,U.S.A.")
print(spacy.displacy.render(doc, style="ent", page="true"))