import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
#nltk.download('averaged_perceptron_tagger')
sentence = 'On the 15th of September, Tim Cook announced that Apple wants to acquire ABC Group from New York for 1 billion dollars.'
tokenized_sent = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokenized_sent)
print(nltk.ne_chunk(tagged_sent))