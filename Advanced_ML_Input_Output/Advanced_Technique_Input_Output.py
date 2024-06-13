#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the Libraies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('insurance_pre.csv')


# In[3]:


dataset=pd.get_dummies(dataset,drop_first=True)


# In[4]:


indep=dataset[['age', 'bmi', 'children','sex_male', 'smoker_yes']]
dep=dataset['charges']


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(indep, dep, test_size = 1/3, random_state = 0)


# In[6]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[7]:


y_train_reshaped = y_train.to_numpy().reshape(-1, 1)
y_test_reshaped = y_test.to_numpy().reshape(-1, 1)

scy=StandardScaler()
y_train=scy.fit_transform(y_train_reshaped)
y_test=scy.transform(y_test_reshaped)


# In[8]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
param_grid = {'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],'max_features': ['sqrt','log2'],'n_estimators':[10,100]}
grid = GridSearchCV(RandomForestRegressor(), param_grid, refit = True, verbose=3,n_jobs=-1)
# fitting the model for grid search
grid.fit(X_train, y_train)


# In[9]:


re=grid.cv_results_
grid_predictions = grid.predict(X_test)
from sklearn.metrics import r2_score
r_score=r2_score(y_test,grid_predictions)
print("The R_score value for best parameter {}:".format(grid.best_params_),r_score)


# In[10]:


table=pd.DataFrame.from_dict(re)


# In[11]:


table


# In[12]:


import pickle
filename="finalized_advancedtechniqueRFIandO).sav"


# In[13]:


pickle.dump(grid,open(filename,'wb'))


# In[14]:


age_input=float(input("Age:"))
bmi_input=float(input("BMI:"))
children_input=float(input("Children:"))
sex_male_input=int(input("Sex Male 0 or 1:"))
smoker_yes_input=int(input("Smoker Yes 0 or 1:"))


# In[15]:


preinput=sc.transform([[age_input,bmi_input,children_input,sex_male_input,smoker_yes_input]])


# In[16]:


preinput


# In[17]:


loaded_model=pickle.load(open("finalized_advancedtechniqueRFIandO).sav", 'rb'))
result=loaded_model.predict(preinput)


# In[18]:


result


# In[19]:


preoutput=scy.inverse_transform([result])


# In[20]:


preoutput

