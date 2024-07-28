import os
from flask import Flask, request, render_template , send_from_directory
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Create the Flask app
app = Flask(__name__)

# Load the data
data_path = 'ipl_data.csv'
ipl = pd.read_csv(data_path)

# Extract unique values for dropdowns
unique_venues = ipl['venue'].unique().tolist()
unique_batting_teams = ipl['bat_team'].unique().tolist()
unique_bowling_teams = ipl['bowl_team'].unique().tolist()
unique_batsmen = ipl['batsman'].unique().tolist()
unique_bowlers = ipl['bowler'].unique().tolist()

# Preprocess the data
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)
X = df.drop(['total'], axis=1)
y = df['total']

# Initialize LabelEncoders
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

# Fit the encoders on the data
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load the trained model
model_path = 'mini7/score_prediction_model.h5'
model = keras.models.load_model(model_path, compile=False)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/index')
def home():
    return render_template('index.html', venues=unique_venues, bat_teams=unique_batting_teams,
                           bowl_teams=unique_bowling_teams, batsmen=unique_batsmen, bowlers=unique_bowlers)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        venue = request.form['venue']
        bat_team = request.form['bat_team']
        bowl_team = request.form['bowl_team']
        batsman = request.form['batsman']
        bowler = request.form['bowler']

        # Encode the categorical data
        venue_encoded = venue_encoder.transform([venue])[0]
        bat_team_encoded = batting_team_encoder.transform([bat_team])[0]
        bowl_team_encoded = bowling_team_encoder.transform([bowl_team])[0]
        batsman_encoded = striker_encoder.transform([batsman])[0]
        bowler_encoded = bowler_encoder.transform([bowler])[0]

        # Prepare the input data
        input_data = np.array([[venue_encoded, bat_team_encoded, bowl_team_encoded, batsman_encoded, bowler_encoded]])
        input_data_scaled = scaler.transform(input_data)

        # Predict the score
        prediction = model.predict(input_data_scaled)
        predicted_score = int(prediction[0,0])

        return render_template('result.html', prediction=predicted_score)

@app.route('/download')
def download():
    return render_template('downloads.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)
