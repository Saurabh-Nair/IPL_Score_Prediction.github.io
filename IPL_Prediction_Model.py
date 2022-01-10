from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

df = pd.read_csv('New_IPL.csv')

df.head()

columns_to_remove = ['match_id','player_dismissed','innings','Innings_ID','runs','runs_off_bat','striker', 'bowler','extras','season', 'non_striker','wicket_type','other_wicket_type','other_player_dismissed']
df.drop(labels = columns_to_remove,axis=1,inplace=True)

#df["player_dismissed"] = df["player_dismissed"].fillna(0)
#df["player_dismissed"]=df["player_dismissed"].apply(lambda x: 1 if x!=0 else 0)

#df['venue'].unique()

#datatoexcel =  pd.ExcelWriter("FromPython.xlsx", engine='xlsxwriter')

#df.to_excel(datatoexcel, sheet_name='Sheet1')

#datatoexcel.save()

consistent_teams = ['Sunrisers Hyderabad','Royal Challengers Bangalore','Mumbai Indians','Kolkata Knight Riders', 'Punjab Kings', 'Delhi Capitals','Chennai Super Kings', 'Rajasthan Royals']
consistent_venue = ['M Chinnaswamy Stadium' ,'Punjab Cricket Association Stadium','Feroz Shah Kotla Stadium','Wankhede Stadium' ,'Sawai Mansingh Stadium','MA Chidambaram Stadium' ,'Eden Gardens','Rajiv Gandhi International Stadium']
df = df [(df['batting_team'].isin(consistent_teams)) & (df['bowling_team'].isin(consistent_teams) )]
df = df[df['venue'].isin(consistent_venue)]
df = df[(df['ball']>=5.0)]

#print(df['batting_team'].unique())
#print(df['bowling_team'].unique())
#print(df['venue'].unique())

from datetime import datetime
df['start_date'] = df['start_date'].apply(lambda x:datetime.strptime(x, '%d-%m-%Y'))

encoded_df = pd.get_dummies(data=df, columns=['batting_team', 'bowling_team','venue'] )

encoded_df = encoded_df[['venue_MA Chidambaram Stadium','venue_Feroz Shah Kotla Stadium','venue_Punjab Cricket Association Stadium',
	   'venue_Eden Gardens','venue_Wankhede Stadium','venue_Sawai Mansingh Stadium',
	   'venue_M Chinnaswamy Stadium','venue_Rajiv Gandhi International Stadium',
       'batting_team_Chennai Super Kings', 'batting_team_Delhi Capitals',
       'batting_team_Punjab Kings', 'batting_team_Kolkata Knight Riders',
       'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
       'batting_team_Royal Challengers Bangalore',
       'batting_team_Sunrisers Hyderabad', 'bowling_team_Chennai Super Kings',
       'bowling_team_Delhi Capitals', 'bowling_team_Punjab Kings',
       'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians',
       'bowling_team_Rajasthan Royals',
       'bowling_team_Royal Challengers Bangalore',
       'bowling_team_Sunrisers Hyderabad', 'start_date', 'ball', 
       'last_5_overs','last_5_wickets','current_runs','total_runs',
       ]]

encoded_df.columns

#datatoexcel =  pd.ExcelWriter("FromPython.xlsx", engine='xlsxwriter')

#encoded_df.to_excel(datatoexcel, sheet_name='Sheet1')

#datatoexcel.save()

X_train = encoded_df.drop(labels='total_runs', axis=1)[encoded_df['start_date'].dt.year <= 2019]
X_test = encoded_df.drop(labels='total_runs', axis=1)[encoded_df['start_date'].dt.year >= 2020]

Y_train = encoded_df[encoded_df['start_date'].dt.year <= 2019]['total_runs'].values
Y_test = encoded_df[encoded_df['start_date'].dt.year >= 2020]['total_runs'].values

X_train.drop(labels='start_date', axis=True, inplace=True)
X_test.drop(labels='start_date', axis=True, inplace=True)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.metrics import accuracy_score
import numpy as np 

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

from sklearn.ensemble import RandomForestRegressor
lin = RandomForestRegressor(n_estimators=100,max_features=None)
lin.fit(X_train,Y_train)

prediction = lin.predict(X_test)
print('Mean Squared Error :', metrics.mean_squared_error(Y_test, prediction))
print('Mean Squared Error :', metrics.mean_squared_error(Y_test, prediction))
print(accuracy_score(Y_test, prediction))

filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
 

 