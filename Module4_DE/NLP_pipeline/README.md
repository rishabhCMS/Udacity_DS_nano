# NLP Pipeline


## How NLP Pipelines work?

The 3 stages of an NLP pipeline are: Text Processing > Feature Extraction > Modeling.

- Text Processing: Take raw input text, clean it, normalize it, and convert it into a form that is suitable for feature extraction.

- Feature Extraction: Extract and produce feature representations that are appropriate for the type of NLP task you are trying to accomplish and the type of model you are planning to use.

- Modeling: Design a statistical or machine learning model, fit its parameters to training data, use an optimization procedure, and then use it to make predictions about unseen data.
This process isn't always linear and may require additional steps.

## Stage 1: Text Processing

- **Cleaning** to remove irrelevant items, such as HTML tags
- **Normalizing** by converting to all lowercase and removing punctuation
- **Splitting** text into words or tokens
- **Removing** words that are too common, also known as stop words
- **Identifying** different parts of speech and named entities
- **Converting** words into their dictionary forms, using stemming and lemmatization

### Cleaning:

````python
#fetching the wen-page using the requests library

# import statements
import requests
from bs4 import BeautifulSoup

# fetch web page
r = requests.get('https://www.udacity.com/courses/all')

soup = BeautifulSoup(r.text, 'lxml')

courses = []
for summary in summaries:
    # append name and school of each summary to courses list
    name=summary.find('h2').get_text()
    school=summary.find('h3').get_text()
    courses.append((name, school))
#     print(school)
    
````

### Normalization

````python
# Convert to lowercase
text = text.lower()

import re
text = re.sub(r'[^a-zA-Z0-9]',' ', text)
print(text)
````

### Tokenization

````python
import nltk

#word tokenize
nltk.word_tokenize(text)

#sentence tokenize
nltk.sent_tokenize(text
````

### Stop Words removal

````python
from nltk.corpus import stopwords

print(stopwords.words('english")
````

### Part of Speech Tagging (POS tagging)

````python
from nltk import pos_tag

sentence_tokenize = nltk.work_tokenize(text)

post_tag(sentence_tokenize)
````

### Named Entity Recognition (NER)
````python
from nltk import ne_chunk
````

### Stemming and Lemmatization

````python 

from nltk.stemmer.porter import PoretrStemmer

from nltk.stem.wordnet import WordNetLemmatizer

https://youtu.be/zKYEvRd2XmI
````
## Stage II: Feature Extraction

**How do you represent text for ML purposes. below are a few techniques to do that**

### Bag of words

### TF-IDF (termfrequncy and Inverse Document frequency)


## Stage III: Modelling
