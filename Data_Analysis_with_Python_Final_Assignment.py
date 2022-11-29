#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'


# In[3]:


df = pd.read_csv(file_name)


# In[4]:


df.head()


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[14]:


df.drop(["id"], axis=1, inplace=True)


# In[15]:


df.drop(["Unnamed: 0"], axis=1, inplace=True)


# In[16]:


df.describe()


# In[17]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[18]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[19]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[20]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[21]:


df['floors'].value_counts().to_frame()


# In[22]:


sns.boxplot(x="waterfront", y="price", data=df)


# In[23]:


sns.regplot(x="sqft_above", y="price", data=df)


# In[26]:


# sqfr_above feature is positively correlated with price.


# In[25]:


df.corr()['price'].sort_values()


# In[27]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# In[29]:


lm.fit(df[['sqft_living']], df[['price']])
lm


# In[34]:


lm.coef_


# In[37]:


lm.intercept_


# In[59]:


lm.predict(X)


# In[63]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     


# In[ ]:





# In[70]:


Z = df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     ]


# In[71]:


lm.fit(Z, df['price'])


# In[72]:


Y_hat = lm.predict(Z)


# In[76]:


lm.fit(Z, df['price'])


# In[77]:


print('The R-square is: ', lm.score(Z, df['price']))


# In[79]:


pr=PolynomialFeatures(degree=2)
pr


# In[80]:


Z_pr=pr.fit_transform(Z)


# In[81]:


Z.shape


# In[82]:


Z_pr.shape


# In[83]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[84]:


pipe=Pipeline(Input)
pipe


# In[102]:


Z = Z.astype(float)
pipe.fit(Z,df['price'])


# In[106]:


ypipe=pipe.predict(Z)
ypipe


# In[118]:


ypipe.shape


# In[123]:


lm.fit(Z, df['price'])


# In[126]:


lm.score(Z, df['price'])


# In[127]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[128]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[129]:


from sklearn.linear_model import Ridge


# In[130]:


RidgeModel=Ridge(alpha=0.1)


# In[133]:


RidgeModel.fit(x_train, y_train)


# In[134]:


RidgeModel.score(x_test, y_test)


# In[ ]:




