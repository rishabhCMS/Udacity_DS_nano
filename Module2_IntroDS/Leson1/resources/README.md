
# Lesson 1 : The Data Science Process

## The CRISP-DM Process ( Cross Industry Process for Data Mining)

    1. Business Understanding
    2. Data Understanding
    3. Data Preperation
    4. Data Modelling
    5. Model Evaluation
    6. Model Deployment
    
## 1. Business Understanding 

Business Understanding means, depending on your Industry domian you are working in, what questions you want to address.
For example:

    - How do we acquire new customers?
    - Does a new treatment perform better than an existing treatment?
    - How can improve communication?
    - How can we improve travel?
    - How can we better retain information?
    
## 2. Understanding Data

    - Either you have the data and then you figure out what questions you wan to answer
    - Or you have the questions you want to answer and then get relevant data

### Python Tricks learned 

```js
# finding non-null columns
df.columns[df.isnull().mean()==0]

#df.query and df[df[col] >10]
df.query("Col_name == some_vale") # returns rows that satisfy the "Col_name = some_vale"

df[df[Col_name] == "some_value"]

#reset_index
df.reset_index()

#rename columns
df.rename(columns = {"col_name":"col_name_changed"}, inplace=True)

#set_index
sd.set_index('col_name')

#plots hist for all columns
df.hist()

#sns heatmap takes in the corr matrix
sns.heatmap(df.corr(), annot=True, fmt=".2f");

#drop nans, how = "all" wants all entries in col(axis= 1) /row(axis=0) to be nan
# how = "any" wants any entry in the col(axis=1)/row(axis=0) to be a nan for removal

small_dataset.dropna(axis=0, how='all')
small_dataset.dropna(subset=['col3'], how='any')

#imputing null values
fill_mean = lambda col: col.fillna(col.mean())
new_df.apply(fill_mean, axis=0)

# select cols of a particlular type
df.select_dtypes(include=['object'])

#dummy encode NaN values as their own dummy coded column using the dummy_na argument
dummy_cols_df = pd.get_dummies(dummy_var_df['col1'], dummy_na=True)
```

## 3. Prepare Data

  This is commonly denoted as 80% of the process. 
  You saw this especially when attempting to build a model to predict salary, 
  and there was still much more you could have done. 
  From working with missing data to finding a way to work with categorical variables, 
  and we didn't even look for outliers or attempt to find points we were especially poor at predicting. 
  There was ton more we could have done to wrangle the data, but you have to start somewhere, 
  and then you can always iterate.
  
  
## 4. Model Data

  We were finally able to model the data, but we had some back and forth with step 3. before we were able to build a model that had okay performance. There still may be changes that could be done to improve the model we have in place. From additional feature engineering to choosing a more advanced modeling technique, we did little to test that other approaches were better within this lesson.
  
## 5. Evaluate Model

  Results are the findings from our wrangling and modeling. They are the answers you found to each of the questions.
 
## 6. Deploy Model

  Deploying can occur by moving your approach into production or by using your results to persuade others within a company to act on the results. Communication is such an important part of the role of a data scientist.
