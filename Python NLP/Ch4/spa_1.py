import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp("Berlin is the capital of Germany")
doc.ents

print(doc.ents[1],doc.ents[1].label_)