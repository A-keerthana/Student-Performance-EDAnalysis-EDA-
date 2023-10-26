#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# In[6]:


df = pd.read_csv('student_extended_ml_dataset2.csv')


# In[7]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.describe()


# In[11]:


df.info()


# In[12]:


#for checking the missing values
print(df.isnull().sum())

#for checking the duplicates
print(df.duplicated().sum())

#for checking the outliers using box plot
plt.boxplot(df['IQ'])
plt.show()


# In[18]:


#performing univariate analysis
Math_Marks = df['Math_Marks']

#creating a histogram
plt.hist(Math_Marks)
plt.xlabel('Math Marks')
plt.ylabel('Frequency')
plt.show()

#Creating a box plot
plt.boxplot(x=Math_Marks)
plt.xlabel('Math Marks')
plt.show()

#creating a box plot using seaborn
sns.boxplot(x=Math_Marks)
plt.xlabel('Math Marks')
plt.show()


# In[19]:


#performing Bivariant Analysis
Math_Marks = df['Math_Marks']
Chemistry_Marks = df['Chemistry_Marks']

#creating a scatter plot
plt.scatter(Math_Marks, Chemistry_Marks)
plt.xlabel('Math Marks')
plt.ylabel('Chemistry Marks')
plt.show()

#creating a line plot
plt.plot(Math_Marks, Chemistry_Marks)
plt.xlabel('Math Marks')
plt.ylabel('Chemistry Marks')
plt.show()


# In[21]:


#performing Multivariant analysis
selected_vars = ['Math_Marks', 'Chemistry_Marks', 'Physics_Marks']

#creating a heatmap
sns.heatmap(df[selected_vars].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#creating a pair plot
sns.pairplot(df[selected_vars])
plt.show()


# In[31]:


#creating a new feature [feature engineering]
df = pd.read_csv('student_extended_ml_dataset2.csv')

Math_Marks = df['Math_Marks']
Chemistry_Marks = df['Chemistry_Marks']
Physics_Marks = df['Physics_Marks']

df['Total_Marks'] = Math_Marks + Chemistry_Marks + Physics_Marks
print(df.head())


# In[46]:


#model selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
X = df[['IQ', 'Math_Marks', 'Chemistry_Marks', 'Physics_Marks']]
y = df['Total_Marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{model.__class__.__name__}: {mse}')
    
best_model = LinearRegression()
best_model.fit(X_train, y_train)


# In[47]:


#Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = model.score(X_test, y_test)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r_squared}')

