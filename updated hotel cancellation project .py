#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries 
# 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r'/Users/snehagupta/Desktop/Placement - THE GOAL/ML-PROJECTS /HOTELS CANCELLATION PROJECT/hotel_bookings.csv')


# In[3]:


type(df)


# In[4]:


df.head(4)


# In[5]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# ## Data Cleaning

# In[6]:


df.isnull()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(['agent','company'],axis=1, inplace=True) ## to remove the agent and company column from the data set 


# In[9]:


(df['country'].isnull().sum()/ df.shape[0])*100  ## concludes the proportion of the country column in the data set 


# In[10]:


df['country'].value_counts() # gives the frequency value in the country column in dec order 


# In[11]:


df['country'].value_counts().index[0] # gives the most frequent value 


# In[12]:


df['country'].fillna(df['country'].value_counts().index[0],inplace= True)   ## fills all the NaN values in the 'country' column with the most frequent country PRT


# In[13]:


df.fillna(0,inplace=True) # change all the NAN values with 0 in complete data


# In[14]:


df.isnull().sum()


# In[15]:


filter1 = (df['children']==0)& (df['adults']==0)& (df['babies']==0)
df[filter1]


# In[16]:


data = df[~filter1]


# In[17]:


data.shape


# In[ ]:





# ## EDA

# ### % of people according to the country who has not canceled their booking 

# In[18]:


(data[data['is_canceled']==0]['country'].value_counts()/75011)* 100   


# In[19]:


len(data[data['is_canceled']==0])


# In[ ]:





# In[20]:


country_wise_data = data[data['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns = ['country','No. of Guests']
country_wise_data


# In[21]:


#!pip install plotly


# In[22]:


#!pip install chart_studio


# In[23]:


import plotly
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs , init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[24]:


import plotly.express as px


# In[25]:


map_guest = px.choropleth(country_wise_data, 
                          locations = country_wise_data['country'] , 
                          color = country_wise_data['No. of Guests'], 
                          hover_name = country_wise_data['country'], 
                          title = 'Home country of Guests')


# In[26]:


map_guest.show()


# In[ ]:





# ### How much do Guest pay for a room per night 

# In[ ]:





# In[27]:


data2 = data[data['is_canceled']==0]


# In[28]:


data2


# In[29]:


data.columns


# In[30]:


# seaborn boxplot:

plt.figure(figsize= (12,8))
sns.boxplot(x='reserved_room_type' , y = 'adr' , hue = 'hotel' ,data = data2)
plt.xlabel('room types')
plt.ylabel('price(EUR)')
plt.title('price of rooms per night and per person')


# In[31]:


# Result :  1) 
#           2)


# ### ANALYSING DEMAND OF HOTELS

# In[ ]:





# In[32]:


data['hotel'].unique()


# In[33]:


data_resort = data[(data['hotel']== 'Resort Hotel') & (data['is_canceled']==0)]
data_cityhotel = data[(data['hotel']== 'City Hotel') & (data['is_canceled']==0)]


# In[34]:


data_resort.head(3)


# In[35]:


rush_resort = data_resort['arrival_date_month'].value_counts().reset_index()   ##. for resorts 
rush_resort.columns = ['Months', 'No of Guests']
rush_resort


# In[ ]:





# In[36]:


rush_cityhotel = data_cityhotel['arrival_date_month'].value_counts().reset_index()  ## for city hotels
rush_cityhotel.columns = ['Months', 'No of Guests']
rush_cityhotel


# In[ ]:





# In[37]:


final_rush = rush_resort.merge(rush_cityhotel,on = 'Months')


# In[38]:


final_rush.columns = ['Months','No_of_guests_in_resorts' ,'No_of_guests_in_cityhotel' ]
final_rush


# In[ ]:





# In[39]:


#!pip install sorted-months-weekdays
#!pip install sort_dataframeby_monthorweek


# In[40]:


import sort_dataframeby_monthorweek as sd


# In[41]:


final_rush = sd.Sort_Dataframeby_Month(final_rush,'Months')


# In[42]:


final_rush.columns


# In[43]:


px.line(data_frame= final_rush, x = 'Months', y = ['No_of_guests_in_resorts' ,'No_of_guests_in_cityhotel'])


# In[ ]:





# ### Analyze whether bookings were made only for weekdays or weekends or both

# In[44]:


data.columns


# In[45]:


data


# In[46]:


def week_function(row):
    feature1 = 'stays_in_weekend_nights'
    feature2 = 'stays_in_week_nights'
    
    if row[feature2]==0 and row[feature1]>0:
        return 'stay_just_weekend'
    elif row[feature2]>0 and row[feature1]==0:
        return 'stay_just_weekdays'
    elif row[feature1]>0 and row[feature2]>0:
        return 'stay_both_weekend_weekdays'
    else:
        return 'undefined_data'


# In[47]:


data2['weekdays_or_weekends']=data2.apply(week_function,axis=1)


# In[48]:


data2['weekdays_or_weekends'].value_counts()


# In[ ]:





# In[49]:


data2=sd.Sort_Dataframeby_Month(data2,'arrival_date_month')


# In[50]:


group_data=data2.groupby(['arrival_date_month','weekdays_or_weekends']).size()


# In[51]:


group_data=data2.groupby(['arrival_date_month','weekdays_or_weekends']).size().unstack().reset_index()


# In[52]:


group_data


# In[53]:


sorted_data=sd.Sort_Dataframeby_Month(group_data,'arrival_date_month')


# In[54]:


sorted_data


# In[55]:


sorted_data.set_index('arrival_date_month',inplace=True)


# In[56]:


sorted_data.plot(kind='bar',stacked=True,figsize=(15,10))


# In[ ]:





# ### Creating some useful feature for applying ML model

# In[57]:


data2.columns


# In[58]:


def family(row):
    if (row['adults']>0) & (row['children']>0 or row['babies']>0):
        return 1 
    else:
        return 0


# In[59]:


data['is_family']=data.apply(family,axis=1)


# In[60]:


data['total_customer']= data['adults']+data['babies']+data['children']


# In[61]:


data['total_nights']=data['stays_in_week_nights']+data['stays_in_weekend_nights']


# In[62]:


data.head(3)


# In[63]:


data['deposit_type'].unique()


# In[64]:


dict1={'No Deposit':0, 'Non Refund':1, 'Refundable': 0}


# In[65]:


data['deposit_given']=data['deposit_type'].map(dict1)


# In[66]:


data.columns


# In[67]:


data.drop(columns=['adults', 'children', 'babies', 'deposit_type'],axis=1,inplace=True)


# In[68]:


data.columns


# In[69]:


data.head(6)


# ###  Applying feature encoding on categorical data

# In[ ]:





# In[70]:


data.dtypes


# In[71]:


cate_features = [col for col in data.columns
                    if data[col].dtype == 'object']
                      


# In[72]:


num_features = [col for col in data.columns
                    if data[col].dtype != 'object']


# In[73]:


cate_features


# In[74]:


num_features


# In[75]:


data_cat=data[cate_features]


# In[76]:


data.groupby(['hotel'])['is_canceled'].mean().to_dict()


# In[77]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[78]:


data_cat['cancellation']= data['is_canceled']


# In[79]:


data_cat.head()


# In[80]:


cols=data_cat.columns


# In[81]:


cols


# In[82]:


for col in cols:
    dict2= data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict2)
     


# In[83]:


data_cat.head()


# In[ ]:





# ### To Handle Outliers

# In[84]:


dataframe=pd.concat([data_cat,data[num_features]],axis=1)


# In[85]:


dataframe.columns


# In[86]:


sns.distplot(dataframe['lead_time'])


# In[87]:


def handle_outlier(col):
    dataframe[col] = np.log1p(dataframe[col])


# In[88]:


handle_outlier('lead_time')


# In[89]:


sns.distplot(dataframe['lead_time'])


# In[90]:


# For average daily rate (adr)


# In[91]:


sns.distplot(dataframe['adr'])


# In[ ]:





# In[ ]:





# In[92]:


dataframe[dataframe['adr']<0]


# In[93]:


handle_outlier('adr')


# In[94]:


dataframe['adr'].isnull().sum()


# In[95]:


### now why this missing value , as we have already deal with the missing values..'
### bcz we have negative value in 'adr' feature as '-6.38'  ,& if we apply ln(1+x) , we will get 'nan'
## bcz log wont take negative values.


# In[96]:


sns.distplot(dataframe['adr'].dropna())


# In[ ]:





# ### Selecting important feature using Co-relation and Univarite analysis 

# In[ ]:





# In[97]:


sns.FacetGrid(data,hue='is_canceled', xlim = (0,500)).map(sns.kdeplot,'lead_time',shade=True).add_legend() 
# KDE PLOT -- a method for visualizing the distribution of observations in a dataset, analogous to a histogram


# In[98]:


corr = dataframe.corr()


# In[99]:


corr


# In[100]:


corr['is_canceled'].sort_values(ascending=False)


# In[101]:


corr['is_canceled'].sort_values(ascending=False).index


# In[102]:


#. HIGH CORELATION - shows overfitting 
#  LOW CORELATION - shows low accuracy in ML model


# In[ ]:





# In[103]:


## Now we drop the features which have high corelation and a very low corelation 


# In[104]:


features_to_drop = ['reservation_status','cancellation','arrival_date_year','arrival_date_week_number',
                    'stays_in_weekend_nights','arrival_date_day_of_month','reservation_status_date']


# In[105]:


dataframe.drop(features_to_drop,axis=1,inplace=True)


# In[106]:


dataframe.shape


# ### Applying tecniques for of Feature importance

# In[ ]:





# In[107]:


dataframe.isnull().sum()


# In[108]:


dataframe.dropna(inplace=True)


# In[109]:


dataframe


# In[ ]:





# In[110]:


## separate dependent & independent features


# In[111]:


x = dataframe.drop('is_canceled', axis= 1)


# In[112]:


y = dataframe['is_canceled']


# In[ ]:





# In[113]:


from sklearn.linear_model import Lasso                   
from sklearn.feature_selection import SelectFromModel


# In[114]:


########. WHY LASSO   #################


# In[115]:


## Least Absolute Shrinkage and Selection Operator
## can shrink some coefficients to exactly zero.
## This property helps automatically select the most relevant features by effectively discarding irrelevant ones.
## By reducing the number of features, it mitigates the risk of overfitting and improves the model's 
## generalizability on unseen data.
## The bigger the alpha the less features that will be selected.



# In[116]:


# Lasso(alpha=0.005)


# In[117]:


feature_sel_model = SelectFromModel(Lasso(alpha=0.005))


# In[118]:


feature_sel_model.fit(x,y)


# In[119]:


feature_sel_model.get_support()


# In[120]:


cols = x.columns


# In[ ]:





# In[121]:


# print the number of selected features

selected_feature = cols[feature_sel_model.get_support()]


# In[122]:


selected_feature


# In[123]:


x=x[selected_feature]


# In[124]:


x


# In[125]:


y


# In[126]:


np.sum(y ==1)


# In[ ]:





# ### APPLYING LOGISTIC REGRESSION 

# In[127]:


from sklearn.model_selection import train_test_split


# In[128]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[129]:


x_train.shape


# In[130]:


from sklearn.linear_model import LogisticRegression


# In[131]:


logreg = LogisticRegression()


# In[132]:


logreg.fit(x_train,y_train)


# In[133]:


pred=logreg.predict(x_test)


# In[134]:


pred


# In[ ]:





# In[135]:


# Accuracy: (TP + TN) / (TP + TN + FP + FN)
# Precision: TP / (TP + FP)
# Recall (Sensitivity): TP / (TP + FN)
# F1 Score: 2 * (Precision * Recall) / (Precision + Recall)


# In[136]:


from sklearn.metrics import confusion_matrix


# In[137]:


conf_mat = confusion_matrix(y_test,pred)


# In[138]:


conf_mat


# In[139]:


accurecy = (conf_mat[0][0] + conf_mat[1][1]) /conf_mat.sum()


# In[140]:


accurecy 


# In[141]:


Precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])


# In[142]:


Precision 


# 
# ### Cross Validation 

# In[143]:


from sklearn.model_selection import cross_val_score


# In[144]:


score = cross_val_score(logreg,x,y,cv=10)


# In[145]:


score.mean()


# In[ ]:





# ### Applying multiple algorithms of ML:

# In[ ]:





# In[146]:


# At the time of overfitting we do pre- pruning  and post pruning 


# In[147]:


from sklearn.naive_bayes  import GaussianNB
from sklearn.linear_model  import LogisticRegression
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.tree  import DecisionTreeClassifier


# In[150]:


models= []
models.append((" Logistic Regression ",LogisticRegression()))
models.append(("Naive Bays ",GaussianNB()))
models.append(("Random Forest ",RandomForestClassifier()))
models.append((" Decision Tree",DecisionTreeClassifier()))
models.append(("KNN ",KNeighborsClassifier()))


# In[152]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

for name, model in models:
    print(f"Model: {name}")
    
    # Fit the model
    model.fit(x_train, y_train)
    
    # Make predictions
    predictions = model.predict(x_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # Accuracy
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.2f}")
    
    # Precision
    precision = precision_score(y_test, predictions)
    print(f"Precision: {precision:.2f}")
    
    # Recall
    recall = recall_score(y_test, predictions)
    print(f"Recall: {recall:.2f}")
    
    # F1-Score
    f1 = f1_score(y_test, predictions)
    print(f"F1-Score: {f1:.2f}")
    
    print("\n")



# In[ ]:



         
          


# In[ ]:





# In[ ]:





# In[ ]:




