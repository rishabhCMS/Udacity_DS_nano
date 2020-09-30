# ETL (Extract Transform & Load Data) Pipelines

Big Data links

1. https://www.udacity.com/course/intro-to-hadoop-and-mapreduce--ud617

2. https://www.udacity.com/course/deploying-a-hadoop-cluster--ud1000

3. https://www.udacity.com/course/real-time-analytics-with-apache-storm--ud381

4. https://www.udacity.com/course/big-data-analytics-in-healthcare--ud758

## 1. Extract data from different sources such as:



## 2. Transform data


## 3. Load

## ETL Pipeline

** to find the encoding of a file**
```` python
# import the chardet library
import chardet 

# use the detect method to find the encoding
# 'rb' means read in the file as binary
with open("mystery.csv", 'rb') as file:
    print(chardet.detect(file.read()))
````
