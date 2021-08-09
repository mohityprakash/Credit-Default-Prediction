#!/usr/bin/env python
# coding: utf-8

# Credit Default Prediction

# Libraries used-
# NumPy,
# Pandas,
# Seaborn,
# SciPy,
# Scikit-learn

# In[1]:


import numpy as np 
import pandas as pd 

# import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error


# **Import the Data

# In[2]:


pd.set_option('max_columns', None)
df = pd.read_csv("/Users/mohit/Downloads/Case_Study_for_Moody's/credit/data.csv")


# In[3]:


print(df.columns)


# Renaming the columns for clarity using Word file descriptions

# In[4]:


df.rename(columns = {'X1':'Credit_Amt','X2':'Gender','X3':'Education','X4':'Marital_st','X5':'Age','X6':'RepaySt1','X7':'RepaySt2','X8':'RepaySt3','X9':'RepaySt4','X10':'RepaySt5','X11':'RepaySt6','X12':'BillSt1','X13':'BillSt2','X14':'BillSt3','X15':'BillSt4','X16':'BillSt5','X17':'BillSt6','X18':'PrevPay1','X19':'PrevPay2','X20':'PrevPay3','X21':'PrevPay4','X22':'PrevPay5','X23':'PrevPay6'}, inplace=True)


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.info()


# **Checking for null values

# In[8]:


df.isnull().sum()


# In[9]:


print(df.apply(lambda col: col.unique()))


# **Initial Cleaning
# 
# As we can see, Education & Marital Status have some values which are not described in Data Dictionary, so we just put them into others.
# 
# RepaySt1-6 also have -2 & -1 which are not described in data dictionary. I assume that these are payments in advance and do not change anything considering they are important variables.

# In[10]:


df["Education"]= df["Education"].replace([0,5,6],4)
df["Marital_st"]= df["Marital_st"].replace([0],3)


# **Initial Data Exploration

# In[11]:


plt.subplots(figsize=(20,18))
plt.subplot(221)

df['Gender'].hist()
plt.xlabel('Gender')
plt.title('Gender (1= Male, 2= Female)')

plt.subplot(222)
df['Education'].hist()
plt.xlabel('Education')
plt.title('Education')

plt.subplot(223)
df['Marital_st'].hist()
plt.xlabel('Marital_st')
plt.title('Marital_st')

plt.subplot(224)
df['Age'].hist()
plt.xlabel('Age')
plt.title('Age Distribution')


plt.show()


# In[12]:


df['age_cat'] = pd.cut(df['Age'], range(20, 80, 10), right=False)


# **Plotting demographic features against Credit Default Status

# In[13]:


fig, ax = plt.subplots(1,4)
fig.set_size_inches(20,5)
fig.suptitle('Defaulting by relative numbers given each class, for various demographics')

d = df.groupby(['Y', 'Gender']).size().unstack(level=1)
d = d / d.sum()
d.plot(kind='bar', ax=ax[0])

d = df.groupby(['Y', 'Marital_st']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[1])

d = df.groupby(['Y', 'age_cat']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[2])

d = df.groupby(['Y', 'Education']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[3])


# Nothing concrete can be said from this. Just that men & high school educated people are slightly more likely to default. But the differences are not high.

# In[14]:


g = sns.FacetGrid(df, col='Gender', hue='Y', height=5)
g.map(plt.hist, 'Age', alpha=0.7, bins=9) 
g.add_legend()
plt.show()


# Despite having much more number of women (Gender =2) in age group 30-40, number of defaulter are almost the same. Fair to say that menin age group 30-40 are likely to default more than women in same age group.

# In[15]:


g = sns.FacetGrid(df, col='Gender', hue='Y', height=5)
g.map(plt.hist, 'Marital_st', alpha=0.7) 
g.add_legend()
plt.show()


# In[16]:


g = sns.FacetGrid(df, col='Marital_st', hue='Y', height=5)
g.map(plt.hist, 'Gender', alpha=0.7) 
g.add_legend()
plt.show()


# Nothing much can be deduced from marital status just that men are more likely to default than women irrespective of marital status.

# In[17]:


g = sns.FacetGrid(df, col='Gender', row='Marital_st', hue='Y', height=4)
g.map(plt.hist, 'Age',bins=10) 
g.add_legend()
plt.show()


# In[18]:


def cols(prefix):
    return [prefix+str(x) for x in range(1,7)]

repay_cols = cols('RepaySt')
figure, ax = plt.subplots(2,3)
figure.set_size_inches(18,8)


for i in range(len(repay_cols)):
    row,col = int(i/3), i%3

    d  = df[repay_cols[i]].value_counts()
    x = df[repay_cols[i]][(df['Y']==1)].value_counts()
    ax[row,col].bar(d.index, d, align='center', color='red')
    ax[row,col].bar(x.index, x, align='center', color='yellow', alpha=0.7)
    ax[row,col].set_title(repay_cols[i])


# In[19]:


fig, ax = plt.subplots(3,2)
fig.set_size_inches(22,17)
fig.suptitle('Defaulting by relative numbers given each class, for various Repayment Status')

d = df.groupby(['Y', 'RepaySt1']).size().unstack(level=1)
d = d / d.sum()
d.plot(kind='bar', ax=ax[0,0])

d = df.groupby(['Y', 'RepaySt2']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[0,1])

d = df.groupby(['Y', 'RepaySt3']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[1,0])

d = df.groupby(['Y', 'RepaySt4']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[1,1])

d = df.groupby(['Y', 'RepaySt5']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[2,0])

d = df.groupby(['Y', 'RepaySt6']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[2,1])


# From these two Repayment Status plots, it is fair to say if in the first month i.e. April, if a person has payment delay for two months or more, then he's around 70% likely to default. But as we proceed to later months, people who have delays for more than six months are more likely to default.

# In[20]:


correlation = df.corr()
correlation["Y"].sort_values(ascending=False)


# In[21]:


plt.figure(figsize=(20,15))
a = sns.heatmap(correlation, cmap='YlGnBu', square=True, annot=True, fmt='.2f', linecolor='green')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()


# In[22]:


X = df.drop(['Y'],axis=1)
X.corrwith(df['Y']).plot.bar(figsize = (20, 5), title = "Correlation with Y", 
                                        fontsize = 15,rot = 90, grid = True)
plt.show()


# One hot encoding: We need to create dummies of categorical variables which don't have levels in it for eg Gender. 1 or 2 doesn't mean higher or lower. It just means two different types.

# In[23]:


df = pd.get_dummies(df,columns=["Gender","Education","Marital_st"])


# In[24]:


df=df.drop(['age_cat'],axis=1)


# In[25]:


print(df.columns)


# In[30]:


df=df.rename(columns = {'Gender_1':'Male','Gender_2':'Female','Education_1':'Grad_School','Education_2':'University','Education_3':'High_School','Education_4':'Others','Marital_st_1':'Married', 'Marital_st_2':'Single','Marital_st_3':'Others1'})


# In[31]:


df


# In[32]:


X1 = df[['Male','Female','Grad_School','University','High_School','Others','Married','Single','Others1']]
X1.corrwith(df['Y']).plot.bar(figsize = (20, 5), title = "Correlation with Y", 
                                        fontsize = 15,rot = 90, grid = True)
plt.show()


# As we can see, demographic features have almost no correlation with default status. Only some relative remarks canbemade i.e. Men, married people are more likely to default than females & single people.

# In[33]:


df_X = df.drop(['Y'], axis=1)
df_y = df.Y

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=10)


model1 = LogisticRegression()
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)

print(classification_report(y_pred, y_test))
print('\nAccuracy Score for Logistic Regression: ', accuracy_score(y_pred,y_test))
scoresLR = cross_val_score( model1, X_train, y_train, cv=10)
print("Mean Logistic Regression CrossVal Accuracy on Train Set= %.2f, with std=%.2f" % (scoresLR.mean(), scoresLR.std() ))

scoresLR1 = cross_val_score( model1, X_test, y_test, cv=10)
print("Mean Logistic Regression CrossVal Accuracy on Test Set= %.2f, with std=%.2f" % (scoresLR.mean(), scoresLR.std() ))


# In[34]:


classifier = RandomForestClassifier(n_estimators=10)
classifier.fit( X_train, y_train )
y_pred = classifier.predict( X_test )

cm = confusion_matrix( y_test, y_pred )
print("Accuracy on Test Set for RandomForest = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresRF = cross_val_score( classifier, X_train, y_train, cv=10)
print("Mean RandomForest CrossVal Accuracy on Train Set= %.2f, with std=%.2f" % (scoresRF.mean(), scoresRF.std() ))
print(classification_report(y_pred, y_test))


# In[35]:


classifier1 = DecisionTreeClassifier(random_state=0)
classifier1.fit( X_train, y_train )
y_pred = classifier1.predict( X_test )

cm = confusion_matrix( y_test, y_pred )
print("Accuracy on Test Set for DecisionTree = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresRF = cross_val_score( classifier1, X_train, y_train, cv=10)
print("Mean DecisionTree CrossVal Accuracy on Train Set= %.2f, with std=%.2f" % (scoresRF.mean(), scoresRF.std() ))
print(classification_report(y_pred, y_test))


# In[36]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier'],
    
    'Score': [0.78, 0.81, 0.72]
                    })


# In[37]:


models


# Logistic regression predicts 78% of the classifications right.
# 
# Random forest classifier predicts 81% of the classifications right.
# 
# Decision Tree classifier predicts 72% of the classifications right.

# In[ ]:




