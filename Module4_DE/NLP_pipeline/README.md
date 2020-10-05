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
