#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


train_data=pd.read_excel('Data_Train_flight.xlsx')


# In[3]:


pd.set_option('display.max_columns',None)


# In[4]:


train_data.shape


# In[5]:


train_data.head()


# In[6]:


train_data.info()


# In[7]:


train_data.dropna(inplace=True)


# In[8]:


print(train_data.isnull().sum())


# In[9]:


train_data['Duration'].value_counts()


# In[10]:


train_data["Journey_day"]=pd.to_datetime(train_data.Date_of_Journey,format='%d/%m/%Y').dt.day


# In[11]:


train_data['Journey_month']=pd.to_datetime(train_data.Date_of_Journey,format='%d/%m/%Y').dt.month


# In[12]:


train_data.head()


# In[13]:


# As we have converted Date_of_Journey into integers, kets drop the column

train_data.drop(['Date_of_Journey'],axis=1,inplace=True)


# In[14]:


#Similar to Date_of_Journey we cvan extract values from Dep_Time

#Extracting hours
train_data['Dep_hour']=pd.to_datetime(train_data['Dep_Time']).dt.hour

#Extracting Minutes
train_data['Dep_min']=pd.to_datetime(train_data['Dep_Time']).dt.minute

#Now e can drop the Dep_Time column
train_data.drop(['Dep_Time'],axis=1,inplace=True)


# In[15]:


train_data.head()


# In[16]:


#lets extract the values rom Arrival_Time as well

#Extracting Hours
train_data['Arrival_hour']=pd.to_datetime(train_data.Arrival_Time).dt.hour

#Extractng Minutes
train_data['Arrival_min']=pd.to_datetime(train_data.Arrival_Time).dt.minute

#Now we can drop the Arrival Time Column
train_data.drop(['Arrival_Time'],axis=1,inplace=True)


# In[17]:


train_data.head()


# In[18]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
     


# In[19]:


#Adding duration_hors and duratiom_min list to dataframe
train_data['Duration_Hours']=duration_hours
train_data['Duration_mins']=duration_mins


# In[20]:


train_data.drop(["Duration"], axis = 1, inplace = True)


# In[21]:


train_data.head()


# 
# Handling Categorical Data
# One can find many ways to handle categorical data. Some of them categorical data are,
# 
# **Nominal data** --> data are not in any order --> **OneHotEncoder** is used in this case
# **Ordinal data** --> data are in order --> **LabelEncoder** is used in this case
# 

# In[22]:


train_data['Airline'].value_counts()


# In[23]:


sns.catplot(y="Price",x='Airline', data=train_data.sort_values('Price',ascending=False),kind='boxen',height=6,aspect=3)
plt.show()


# In[24]:


#Airline being Nominal Categorical feature OneHotEncoding will be used.

Airline=train_data[['Airline']]

Airline=pd.get_dummies(Airline,drop_first=True)

Airline.head()


# In[25]:


train_data['Source'].value_counts()


# In[26]:


#Source vs Price

sns.catplot(y='Price',x='Source',data=train_data.sort_values('Price',ascending=False),kind='boxen',height=5,aspect=5)


# In[27]:


#Source is Nominal Categorial feature. So OneHotEncoding will be used

Source=train_data['Source']

Source=pd.get_dummies(Source,drop_first=True)

Source.head()


# In[28]:


train_data['Destination'].value_counts()


# In[29]:


#Destination being Nominal Categorical feature. We will use OneHotEncoding

Destinantion=train_data['Destination']

Destination=pd.get_dummies(Destinantion, drop_first=True)

Destination.head()


# In[30]:


train_data['Route']


# In[31]:


# Route and Total_Stops represent the same kind of data so one of them can be dropped
#Additional_info columns has most;y no_info mentioned so it can be dropped too

train_data.drop(['Additional_Info','Route'],axis=1,inplace=True)


# In[32]:


train_data['Total_Stops'].value_counts()


# In[33]:


#The Route data is of Ordinal Categorical Type so we will use LabelEncoder
#lets assign keys to the data
train_data.replace({'non-stop':0,'1 stop':1,"2 stops":2,'3 stops':3,'4 stops':4})


# In[34]:


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

train_data['Total_Stops']=label_encoder.fit_transform(train_data['Total_Stops'])


# In[35]:


#Lets concatenate  Airline + Source + Destination+train_data

data_train=pd.concat([Airline,Source,Destination,train_data],axis=1)


# In[36]:


data_train.head()


# In[37]:


data_train.drop(["Airline", "Source", "Destination"],axis=1,inplace=True)


# In[38]:


data_train.head()


# In[39]:


data_train.shape


# # Test Data

# In[40]:


test_data=pd.read_excel('Test_set.xlsx')


# In[41]:


test_data.head()


# In[42]:


test_data.info()


# In[43]:


test_data["Duration"].value_counts()


# In[44]:


test_data.dropna(inplace = True)


# In[45]:


test_data.isnull().sum()


# # EDA

# In[46]:


test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day


# In[47]:


test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month


# In[48]:


train_data.head()


# In[49]:


test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[50]:


# Departure time is when a plane leaves the gate. 
# Similar to Date_of_Journey we can extract values from Dep_Time

# Extracting Hours
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour

# Extracting Minutes
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute

# Now we can drop Dep_Time as it is of no use
test_data.drop(["Dep_Time"], axis = 1, inplace = True)


# In[51]:


test_data.head()


# In[52]:


# Arrival time is when the plane pulls up to the gate.
# Similar to Date_of_Journey we can extract values from Arrival_Time

# Extracting Hours
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour

# Extracting Minutes
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute

# Now we can drop Arrival_Time as it is of no use
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[53]:


test_data.head()


# In[54]:


# Assigning and converting Duration column into list
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[55]:


# Adding duration_hours and duration_mins list to train_data dataframe

test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins


# In[56]:


test_data.drop(["Duration"], axis = 1, inplace = True)


# In[57]:


test_data.head()


# # Handling Categorical Data
# 

# In[58]:


# As Airline is Nominal Categorical data we will perform OneHotEncoding

Airline = test_data[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[59]:


# As Source is Nominal Categorical data we will perform OneHotEncoding

Source = test_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[60]:


# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination = test_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# In[61]:


test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[62]:


test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[63]:


test_data.head()


# In[64]:


data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)


# In[65]:


data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[66]:


data_test.head()


# In[67]:


data_test.shape


# # Feature Selection
# 
# ***Finding the best feature which has a significant effect on target variable***

# In[68]:


data_train.shape


# In[69]:


data_train.columns


# In[70]:


X = data_train.loc[:, ['Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Chennai', 'Delhi', 'Kolkata', 'Mumbai', 'Cochin', 'Delhi', 'Hyderabad',
       'Kolkata', 'New Delhi', 'Total_Stops',
       'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min', 'Arrival_hour',
       'Arrival_min', 'Duration_Hours', 'Duration_mins']]
X.head()


# In[71]:


y=data_train.iloc[:,21]
y.head()


# In[72]:


data_train.head()


# In[73]:


#Lest see the relation between depended and independent variables

plt.figure(figsize = (16,16))
sns.heatmap(train_data.corr(), annot = True, cmap = "RdYlGn")

plt.show();


# In[74]:


#Finding the Important Features
from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[75]:


print(selection.feature_importances_)


# In[76]:


#Plotting thr graph of Imo=portant Features

plt.figure(figsize=(15,10))
feat_importances=pd.Series(selection.feature_importances_,index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()


# In[77]:


#Using Random Forest and fitting the model.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[78]:


from sklearn.ensemble import RandomForestRegressor
reg_rf=RandomForestRegressor()
reg_rf.fit(X_train,y_train)


# In[79]:


y_pred=reg_rf.predict(X_test)


# In[80]:


reg_rf.score(X_train,y_train)


# In[81]:


reg_rf.score(X_test, y_test)


# In[82]:


sns.distplot(y_test-y_pred)
plt.show();


# In[83]:


plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel('y_test')
plt.ylabel('y_pred')


# In[84]:


from sklearn import metrics


# In[85]:


print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[86]:


metrics.r2_score(y_test, y_pred)


# # Hyperparameter Tuning
# 

# In[87]:


from sklearn.model_selection import RandomizedSearchCV


# In[88]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[89]:


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[90]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[91]:


rf_random.fit(X_train,y_train)


# In[92]:


rf_random.best_params_


# In[93]:


prediction=rf_random.predict(X_test)


# In[94]:


plt.figure(figsize=(10,10))
sns.distplot(y_test-prediction)
plt.show()


# In[ ]:





# In[95]:


plt.figure(figsize=(10,10))
plt.scatter(y_test,prediction,alpha=1)
plt.xlabel=("y_test")
plt.ylabel=('y_pred')
plt.show()


# In[96]:


print('MAE:',metrics.mean_absolute_error(y_test,prediction))
print('MSE:',metrics.mean_squared_error(y_test,prediction))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[ ]:




