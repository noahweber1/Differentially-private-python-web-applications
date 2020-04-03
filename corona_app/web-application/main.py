#!/usr/bin/env python
from flask import Flask, render_template, flash, request, jsonify, Markup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_splits
import diffprivlib.models as dp

# initializing constant vars
average_survival_rate = 0
# default traveler constants
DEFAULT_AGE = 30
DEFAULT_SEX = 'F'
import pickle

# logistic regression modeling
lr_model = None

app = Flask(__name__)


@app.before_first_request
def startup():
    global average_survival_rate, lr_model 
    # load saved model from web app root directory
    lr_model = pickle.load(open("gbm_model_dump.p",'rb'))

@app.route("/", methods=['POST', 'GET'])
def submit_new_profile():
    model_results = ''
    if request.method == 'POST':
        selected_age = request.form['selected_age']
        selected_sex = request.form['selected_sex']

        # assign new variables to live data for prediction
        age = int(selected_age)

        sex_M = 1 if selected_gender == 'Male' else 0
        sex_F = 1 if selected_gender == 'Female' else 0
         
        # build new array to be in same format as modeled data so we can feed it right into the predictor
        user = [[age, sex_F, sex_M]]
 
        # add user desinged passenger to predict function
        Y_pred = lr_model.predict_proba(user)
        probability_of_surviving_fictional_character = Y_pred[0][1] * 100

        fig = plt.figure()
        objects = ('Average Survival Rate', 'Fictional Traveler')
        y_pos = np.arange(len(objects))
        performance = [average_survival_rate, probability_of_surviving_fictional_character]

        ax = fig.add_subplot(111)
        colors = ['gray', 'blue']
        plt.bar(y_pos, performance, align='center', color = colors, alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.axhline(average_survival_rate, color="r")
        plt.ylim([0,100])
        plt.ylabel('Survival Probability')
        plt.title('How Did Your Fictional Traveler Do? \n ' + str(round(probability_of_surviving_fictional_character,2)) + '% of Surviving!')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html',
            model_results = model_results,
            model_plot = Markup('<img src="data:image/png;base64,{}">'.format(plot_url)),
            selected_age = selected_age,
            selected_gender = selected_gender)
    else:
        # set default passenger settings
        return render_template('index.html',
            model_results = '',
            model_plot = '',
            selected_age = DEFAULT_AGE,
            selected_gender = DEFAULT_GENDER)

if __name__=='__main__':
	app.run(debug=False)