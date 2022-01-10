# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'first-innings-score-lr-model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        venue = request.form['venue']
        if venue == 'MA Chidambaram Stadium':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif venue == 'Feroz Shah Kotla Stadium':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif venue == 'Punjab Cricket Association Stadium':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif venue == 'Eden Gardens':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif venue == 'Wankhede Stadium':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif venue == 'Sawai Mansingh Stadium':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif venue == 'M Chinnaswamy Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif venue == 'Rajiv Gandhi International Stadium':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]

        
        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Capitals':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Punjab Kings':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif bowling_team == 'Delhi Capitals':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Punjab Kings':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif bowling_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        ball = float(request.form['ball'])
        last_5_wickets = int(request.form['last_5_wickets'])
        current_runs = int(request.form['current_runs'])       
        last_5_overs = int(request.form['last_5_overs'])
        #last_5_wickets = int(request.form['wickets_in_prev_5'])
        
        temp_array = temp_array + [ball, last_5_overs,last_5_wickets,current_runs]
        
        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])
              
        return render_template('result.html', lower_limit = my_prediction-10, upper_limit = my_prediction+10)



if __name__ == '__main__':
	app.run(debug=True)