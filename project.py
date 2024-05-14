import sys
import os
import glob
import re

import pickle
import numpy as np

from flask import Flask, render_template, url_for, request, redirect

app = Flask(__name__)

#subscribe_prediction = pickle.load(open('subscribe_prediction.pkl', 'rb'))
subscribe_rf_classifier = pickle.load(open('subscribe_RF_classifier.pkl', 'rb'))

###############################--- home page ---###################################

@app.route('/')
def home():
	return render_template('subscribe_prediction.html')

@app.route('/server_home')
def server_home():
    return render_template('subscribe_prediction.html')

@app.route('/subscribe', methods = ['GET', 'POST'])
def subscribe():
    if (request.method == 'POST'):

        period = int(request.form['period'])
        user_id = int(request.form['user_id'])

        
        sex = request.form['sex']

        age = int(request.form['age'])
        subscribed_days = int(request.form['subscribed_days'])

        multi_screen = request.form['multi_screen']
        mail_subscribed = request.form['mail_subscribed']

        weekly_mins = float(request.form['weekly_mins'])
        min_mins_daily = float(request.form['min_mins_daily'])
        max_mins_daily= float(request.form['max_mins_daily'])

        max_night_mins_weekly = int(request.form['max_night_mins_weekly'])
        Num_of_videos_watched = int(request.form['Num_of_videos_watched'])

        maximum_inactive_days = float(request.form['maximum_inactive_days'])

        calls_customer_care = int(request.form['calls_customer_care'])
 
        #########################################

        
        
        
        sex_d = ['Female', 'Male']
        sex1 = sex_d.index(sex.title())

        multi_screen_d = ['no', 'yes']
        multi_screen1 = multi_screen_d.index(multi_screen.lower())

        mail_subscribed_d = ['no', 'yes']   
        mail_subscribed1 = mail_subscribed_d.index(mail_subscribed.lower())

        #predicting the subcription of a customer
        '''arr = np.array([year, customer_id, phone_no1, gender1, age, no_of_days_subscribed, multi_screen1, mail_subscribed1, weekly_mins_watched, minimum_daily_mins, maximum_daily_mins, weekly_max_night_mins, videos_watched, maximum_days_inactive, customer_support_calls])
        arr = arr.reshape(1,-1)
        res = subscribe_prediction.predict(arr)'''

        #predicting the subcription of a customer
        arr = np.array([period, user_id, sex1, age, subscribed_days, multi_screen1, mail_subscribed1, weekly_mins, min_mins_daily, max_mins_daily,max_night_mins_weekly, Num_of_videos_watched, maximum_inactive_days, calls_customer_care])
        arr = arr.reshape(1,-1)
        res = subscribe_rf_classifier.predict(arr)
        
        if res==0:
            final="user of user id - "+str(int(user_id))
            final+= ' may renew the subscription'
            return render_template("open.html", n = final)
        elif res==1:
            final = "The user of user id - "+str(int(user_id)) + " maynot be interested to renew the subscription.\n"
            reasons=[]
            if(age<12):
                 reasons.append(">> Age might be a reason for non-renewal.")
            if(age>20 and age<59):
                 reasons.append(">> Majority of adults are not renewing.")
            if(subscribed_days<73):
                 reasons.append(">> Low Co-ordination might be a reason.")
            if(multi_screen=="no"):
                 reasons.append(">> Low usage might be the reason.")
            if(mail_subscribed=="no"):
                 reasons.append(">> Low Information about the customer might be the reason.")
            if(weekly_mins==0):
                 reasons.append(">> Low usage might be the reason.")
            if(Num_of_videos_watched==0):
                 reasons.append(">> Low interest towards the platform.")
            if(maximum_inactive_days>4):
                 reasons.append(">> Not using the platform for reasonable no. of days.")
            if(calls_customer_care==0):
                 reasons.append("least interaction with support team.")
            final+="\n**These are the reasons might behind the non-renewal**.\n"
            final+="\n"
            final += "\n\n".join([ reason for reason in reasons])
            n=final
            return render_template("open.html", n= n.replace("\n", "<br>"))
        else:
            final = res
            return render_template("open.html", n =final)


        

    else:
        return render_template("open.html", n='error')

###################################################################################

if __name__ == "__main__":
	app.run(debug = True)
    
