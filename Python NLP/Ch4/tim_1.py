import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from matplotlib import pyplot as plt

f = open('../Ch4/tim_cook.txt', 'r')
article = f.read()

# Tokenize the article into sentences: sentences ทำให้เข้าใจประโยคเข้าใจโครงสร้างประโยค
sentences = nltk.sent_tokenize(article)

# Tokenize each sentence into words: token_sentences ตัดเป็นคำ ๆ
token_sentences = [nltk.word_tokenize(sent) for sent in sentences] 

# Tag each tokenized sentence into parts of speech: pos_sentences ติด tag name entity แต่ละตัวเป็นอะไร
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]

# Create the named entity chunks: chunked_sentences แบ่งเป็นชิ้น ๆ  ***ถ้าจะโชซกราฟต้องเอา binary=True ออก***
chunked_sentences = nltk.ne_chunk_sents(pos_sentences) #binary=True คือการบอกด้วยว่าอันไหนคือ Name Entity : NE แต่ถ้าเอาออกไปเลยมันจะบอกว่ามี NE แต่ละประเภทอะไรบ้าง
print(chunked_sentences)

# # Test for stems of the tree with 'NE' tags print show tag ***ต้องลบตรงนี้ออกไม่งั้น shoe pie ไม่ได้***
# for sent in chunked_sentences:
#     for chunk in sent:
#         if hasattr(chunk, "label") and chunk.label() == "NE":
#             print(chunk)

# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop นำ tag ตัวซ้ำ ๆ กันเพื่อหา เปอร์เซ็น tag
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1

# Create a list from the dictionary keys for the chart labels: labels สร้าง list สำหรับ Dictionary
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(v) for v in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()