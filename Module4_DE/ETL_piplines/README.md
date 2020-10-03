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

## Missing data

Gradient Boosting Decision trees can handle missing values

| data                                           |
|------------------------------------------------|
| 'Other Industry; Trade and Services?$ab' |
| 'Other Industry; Trade and Services?ceg' |


## Outlier detection

resources

http://scikit-learn.org/stable/modules/outlier_detection.html

https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561

### Tukey rule

````python
# TODO: Calculate the first quartile of the population values for 2016
# HINT: you can use the pandas quantile method 
Q1 = df_2016['population'].quantile(0.25)

# TODO: Calculate the third quartile of the population values for 2016
Q3 = df_2016['population'].quantile(0.75)

# TODO: Calculate the interquartile range Q3 - Q1
IQR = Q3 - Q1

# TODO: Calculate the maximum value and minimum values according to the Tukey rule
# max_value is Q3 + 1.5 * IQR while min_value is Q1 - 1.5 * IQR
max_value = Q3 + 1.5 * IQR
min_value = Q1 - 1.5 * IQR

# TODO: filter the population_2016 data for population values that are greater than max_value or less than min_value
population_outliers = None
population_outliers
````

### annotating text to points

````python
# run the code cell below
x = list(df_2016['population'])
y = list(df_2016['gdp'])
text = df_2016['Country Name']

fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x, y)
plt.title('GDP vs Population')
plt.xlabel('population')
plt.ylabel('GDP')
for i, txt in enumerate(text):
    ax.annotate(txt, (x[i],y[i]))
    ````
