#!/usr/bin/env python
# coding: utf-8

# # DATA CLEANING

# In[260]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from windrose import WindroseAxes
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing, metrics
from sklearn.model_selection import TimeSeriesSplit
from bokeh.plotting import figure, show


# In[261]:


#read csv
df1 = pd.read_csv('data/Location1.csv')
df1.head()


# In[262]:


df1.describe()


# In[263]:


#Check row count - 5 years of hourly data
total_hours = 24*365*5
total_hours == len(df1)


# 1) No. of rows match the no. of hours in 5 years, no missing rows
# 2) Temperature min/max suggests the temperature must be in Fahrenheit. Std Dev seems to be fair since weather can fluctuate through seasons.
# 3) Wind Direction min/max suggests the column must be in Degrees
# 4) Power column in normalized between 0-1 (with 1 being max capacity)

# In[264]:


#Renamimg columns
df1 = df1.rename(columns={'Time':'time','Power':'power'})
df1


# In[221]:


df1.info()


# In[222]:


#change Time column to datetime format
df1 = df1.astype({'time':'datetime64[ns]'})
df1.info()


# In[223]:


#Check for nulls
df1.isna().sum()


# In[224]:


#Check for duplicates
df1[df1.duplicated()]


# # DATA TRANSFORMATION

# In[265]:


#Adding time elements to dataframe.
df1['hour'] = pd.DatetimeIndex(df1['time']).hour
df1['dayofweek'] = pd.DatetimeIndex(df1['time']).weekday
df1['month'] = pd.DatetimeIndex(df1['time']).month
df1['year'] = pd.DatetimeIndex(df1['time']).year


# In[266]:


#Adding seasonal encoding to dataframe
summer = [(df1['month']==6) | (df1['month']==7) | (df1['month']==8),
         (df1['month']<6) | (df1['month']>8)]
summer_values = (1,0)
df1['is_summer'] = np.select(summer,summer_values)

fall = [(df1['month']==9) | (df1['month']==10) | (df1['month']==11),
         (df1['month']<9) | (df1['month']>11)]
fall_values = (1,0)
df1['is_fall'] = np.select(fall,fall_values)

winter = [(df1['month']==12) | (df1['month']==1) | (df1['month']==2),
         (df1['month']<12) | ((df1['month']>2)&(df1['month']<12))]
winter_values = (1,0)
df1['is_winter'] = np.select(winter,winter_values)

spring = [(df1['month']==3) | (df1['month']==4) | (df1['month']==5),
         (df1['month']<3) | (df1['month']>5)]
spring_values = (1,0)

df1['is_spring'] = np.select(spring,spring_values)


# In[267]:


#Adding time of day window identifiers to dataframe (12am-4am, 4am-8am, etc.)
conditions = [
    (df1['hour']>=0) & (df1['hour']<5),(df1['hour']>4) & (df1['hour']<9),(df1['hour']>8) & (df1['hour']<13),
    (df1['hour']>12) & (df1['hour']<17),(df1['hour']>16) & (df1['hour']<21),(df1['hour']>20) & (df1['hour']<=23)]

values = ['00:00-04:00','04:00-08:00','08:00-12:00','12:00-16:00','16:00-20:00','20:00-23:00']
df1['timeofday'] = np.select(conditions, values)

df1


# In[268]:


#Data Aggregation for visualization and plotting in next section
year_grouped = df1.groupby('year')['power'].agg('sum')
month_grouped = loc1_train.groupby('month')['power'].agg('sum')
weekday_grouped = loc1_train.groupby('dayofweek')['power'].agg('sum')
timeofday_grouped = loc1_train.groupby('timeofday')['power'].agg('sum')

wind_month = loc1_train.groupby('month')['windspeed_100m'].agg('sum')
wind_day = loc1_train.groupby('dayofweek')['windspeed_100m'].agg('sum')
wind_time = loc1_train.groupby('timeofday')['windspeed_100m'].agg('sum')

hour_grouped = loc1_train.groupby('hour')['power'].agg('sum')
speed_grouped = loc1_train.groupby('hour')['windspeed_100m'].agg('sum')

weekday_grouped = loc1_train.groupby('dayofweek')['power'].agg('sum')


# In[269]:


#Creating training and test data
train = df1['year']<2021
test = df1['year']>2020
loc1_train = df1[train]
loc1_test = df1[test]


# # DATA VISUALIZATION

# In[270]:


#1. Visualizing wind power with time - including time range slider.
fig = px.line(loc1_train, x=loc1_train['time'],y=loc1_train['power'])
fig.update_layout(xaxis_range = ['2017-01-01','2020-12-31'], title_text = 'Wind Power (2017-2020)')

fig.update_xaxes(rangeslider_visible=True)
fig.show()


# In[271]:


#2. Wind power by Year
plt.plot(year_grouped)
plt.axhline(y=np.nanmean(year_grouped), linestyle='dotted',color='r')
plt.xticks(np.arange(2017,2022,1))
plt.title('Total Wind Power by Year')
plt.xlabel('Year')
plt.ylabel('Total Wind Power')
plt.legend(['Wind Power','mean'], loc="lower left")


# In[272]:


#3. Wind power vs. Wind speed by Month
fig, ax1 = plt.subplots()
ax1.plot(month_grouped)
ax2 = ax1.twinx()
ax2.plot(wind_month,color='g',alpha=0.6,linestyle='dashed')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],[0,1,2,3,4,5,6,7,8,9,10,11,12])
plt.title('Wind Power vs. Wind Speeds by Month')
plt.xlabel('Month')
ax1.set_ylabel('Total Wind Power')
ax2.set_ylabel('Total Wind Speed (m/s)')
fig.legend(['Wind Power','Wind Speed'],loc='lower left', bbox_to_anchor=(0.120,0.105))


# In[274]:


#4. Wind power vs. Wind speed by Day of week
fig, ax1 = plt.subplots()
ax1.plot(weekday_grouped)
ax2 = ax1.twinx()
ax2.plot(wind_day,color='g',alpha=0.6,linestyle='dashed')
plt.xticks([0,1,2,3,4,5,6],['Mon','Tues','Wed','Thur','Fri','Sat','Sun'])
plt.title('Wind Power vs. Wind Speeds by Day of Week')
plt.xlabel('Day of Week')
ax1.set_ylabel('Total Wind Power')
ax2.set_ylabel('Total Wind Speed (m/s)')
fig.legend(['Wind Power','Wind Speed'],loc='upper right', bbox_to_anchor=(0.905,0.885))
# ax1.legend(['Wind Power'], loc="lower left")
# ax2.legend(['Wind Speed'], loc="lower left")


# In[273]:


#5. Wind power vs. Wind speed by Time of day
fig, ax1 = plt.subplots()
ax1.plot(timeofday_grouped)
ax2 = ax1.twinx()
ax2.plot(wind_time,color='g',alpha=0.6,linestyle='dashed')
plt.xticks([0,1,2,3,4,5],['00:00-04:00','04:00-08:00','08:00-12:00','12:00-16:00','16:00-20:00','20:00-23:00'])
plt.title('Wind Power vs. Wind Speeds by Time of Day')
plt.xlabel('Time of Day')
ax1.set_ylabel('Total Wind Power')
ax2.set_ylabel('Total Wind Speed (m/s)')
fig.legend(['Wind Power','Wind Speed'],loc='upper right', bbox_to_anchor=(0.905,0.885))
# ax1.legend(['Wind Power'], loc="lower left")
# ax2.legend(['Wind Speed'], loc="lower left")


# In[240]:


#6. Distribution of wind direction and speed
ax = WindroseAxes.from_ax()
ax.bar(loc1_train['winddirection_100m'],loc1_train['windspeed_100m'], normed=True, opening=0.8,colors=('y','g','b','r','c','m'))
ax.set_title('Directional distirbution of Wind Speed')
ax.set_legend(title='Windspeed in m/s',loc=7)


# # MODELING AND VALIDATION

# In[276]:


#Correlation heatmaps for feature selection
columns = ['temperature_2m','relativehumidity_2m','dewpoint_2m','windspeed_10m','windspeed_100m','winddirection_10m','winddirection_100m','windgusts_10m','is_spring','is_summer','is_fall','is_winter','power']
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(df1[columns].corr(),vmin=-1,vmax=1,annot=True)


# In[277]:


#Correlation subplots to visualize features relationship with power
nrows = 5
ncols = 2
fig, ax = plt.subplots(nrows,ncols, figsize=(20,20),sharey='all')

i=0
while i<8:
    for x in range(nrows):
        for y in range(ncols):
            ax[x,y].scatter(loc1_train[columns[i]],loc1_train['power'])
            ax[x,y].title.set_text(columns[i])
            i+=1


# In[278]:


#1. Statsmodel
import statsmodels.formula.api as smf

model = smf.ols(data=loc1_train, formula="power ~ windspeed_100m + windspeed_10m + windgusts_10m + temperature_2m + winddirection_100m + is_spring + is_summer + is_fall + is_winter")
result = model.fit()
result.summary()


# In[280]:


#Evaluating Model scores
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(
    loc1_train['windspeed_100m'], loc1_train['power'])

# get predictions for training and test power
# training
train_regr = intercept + slope*loc1_train['windspeed_100m']
# test
test_regr = intercept + slope*loc1_test['windspeed_100m']


eval_df = pd.DataFrame({"R2": [metrics.r2_score(loc1_train['power'], train_regr), metrics.r2_score(loc1_test['power'], test_regr)],
                        "MAE": [metrics.mean_absolute_error(loc1_train['power'], train_regr), metrics.mean_absolute_error(loc1_test['power'], test_regr)],
                        "RMSE":[metrics.mean_squared_error(loc1_train['power'],train_regr), metrics.mean_squared_error(loc1_test['power'],test_regr)]},
                       index=['training', 'test'])

eval_df


# In[244]:


plt.scatter(loc1_train['windspeed_100m'], loc1_train['power'], label='Training Data')
plt.scatter(loc1_test['windspeed_100m'], loc1_test['power'], color='#00CC00', label='Test Data')
plt.plot(loc1_train['windspeed_100m'], intercept + slope*loc1_train['windspeed_100m'], color='red', label='Regression Model')
plt.title("power ~ windspeed_100m Regression\n training and test data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()


# In[281]:


#2. Scikit learn Model
tss = TimeSeriesSplit(n_splits = 4)
df2 = df1.set_index('year')
df2.sort_index(inplace=True)

#Selecting features
X = df2.drop(labels=['power','time','timeofday','relativehumidity_2m'], axis=1)
y = df2['power']

#Creating train & test data using TimeSeriesSplit function
for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#Fitting Model
model = LinearRegression()
model.fit(X_train,y_train)

#Predictions
predictions = model.predict(X_test)
predictions


# In[282]:


#Evaluating Model scores
MAE = mean_absolute_error(y_test,predictions)
MSE = mean_squared_error(y_test,predictions)
RMSE = np.sqrt(MSE)

print("MAE: %f" % (MAE))
print("RMSE: %f" % (RMSE))


# In[247]:


y_test_data = y_test.reset_index(drop=True)
pred = pd.Series(predictions)
y_test_data


# In[248]:


pred


# In[249]:


res_df = pd.DataFrame({'y_test_data':y_test_data,
                      'predicted':pred,
                      'res':y_test_data - pred})
res_df


# In[250]:


# residuals = y_test - predictions
# sns.scatterplot(x=y_test,y=residuals)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel("Power")
# plt.ylabel("Residuals")
# plt.title("Residual Plot")
# plt.show()


# In[251]:


#1 Bokeh scatter plot with y_test, predictions and residuals
#2 scatter plot with 2017-2020 showing as train, 2021 showing test prediction

p = figure(width=1000, height=500)

# add a circle renderer with a size, color, and alpha
p.circle(y_test_data.index, y_test_data, size=3, color="blue", alpha=0.5, legend_label='y_test')
p.circle(pred.index, pred, size=3, color="green", alpha=0.5,legend_label='predicted')
p.circle(res_df.index, res_df['res'], size=3, color="black", alpha=0.5,legend_label='residual')

#creating legend
p.legend.location = "top_left"
p.legend.click_policy="hide"

# show the results
show(p)


# In[252]:


y_pred_df = df1[df1['year']<2021]
y_pred_df = y_pred_df.set_index('year')
y_pred_df = y_pred_df.drop(labels=['time','temperature_2m','relativehumidity_2m','dewpoint_2m','windspeed_10m','windspeed_100m','winddirection_10m','winddirection_100m','windgusts_10m','hour','dayofweek','month','is_summer','is_fall','is_winter','is_spring','timeofday'],axis=1)
y_pred_df


# In[ ]:




