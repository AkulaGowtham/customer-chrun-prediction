import pandas as pd
###################################### Age asa reason  #############################################################
# Assuming you have your data in a DataFrame called 'data'
data = pd.read_csv('"C:/Users/GOWTHAM/Downloads/OTTPrediction (2)/OTTPrediction/Subscription_Prediction/subcriptions.csv"') # Replace ... with your data
data.dropna(inplace=True)
# Define age groups
age_groups = {
    'child': (0, 12),
    'teen': (13, 19),
    'adult': (20, 59),
    'old': (60, 100)
}

# Calculate non-renewal percentage for each age group
non_renewal_percentages = {}
for age_group, (min_age, max_age) in age_groups.items():
    age_group_data = data[(data['age'] >= min_age) & (data['age'] <= max_age)]
    total_customers = len(age_group_data)
    if(total_customers==0):
        non_renewal_customers=0
        non_renewal_percentage=0
        non_renewal_percentages[age_group]=non_renewal_percentage
    else:
        non_renewal_customers = len(age_group_data[age_group_data['churn'] == 0])
        non_renewal_percentage = (non_renewal_customers / total_customers) * 100
        non_renewal_percentages[age_group] = non_renewal_percentage

# Print the non-renewal percentages for each age group
for age_group, non_renewal_percentage in non_renewal_percentages.items():
    print(f"Non-renewal percentage for {age_group}: {non_renewal_percentage:.2f}%")


# Define the age categories
age_categories = ['child', 'teen', 'adult', 'old']

# Define the age interval boundaries for each category
age_intervals = [0, 12, 19, 59, data['age'].max()]

# Create the age category column based on intervals
data['age_category'] = pd.cut(data['age'], bins=age_intervals, labels=age_categories, right=False)

# Calculate the churn probabilities for each age category
churn_prob_by_age = data.groupby(['age_category', 'churn']).size().unstack().apply(lambda x: x / x.sum(), axis=1)

# Print the churn probabilities
print("Probability of Churn by Age Category:")
for category in age_categories:
    churn_prob_0 = churn_prob_by_age.loc[category, 0]
    churn_prob_1 = churn_prob_by_age.loc[category, 1]
    print(f"churn_prob_by_age_{category} for churn 0:", churn_prob_0)
    print(f"churn_prob_by_age_{category} for churn 1:", churn_prob_1)



###################################### Days Subscribed as a reason###################################################
# Calculate the IQR for days subscribed
days_subscribed_iqr = data['subscribed_days'].quantile(0.75) - data['subscribed_days'].quantile(0.25)

# Calculate the quartiles for days subscribed
q1 = data['subscribed_days'].quantile(0.25)
q2 = data['subscribed_days'].quantile(0.50)  # Median
q3 = data['subscribed_days'].quantile(0.75)

print("Days Subscribed:")
print(f"Q1: {q1:.2f}")
print(f"Q2 (Median): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"days_subscribed_iqr : {days_subscribed_iqr:.2f}")

# Define the intervals for days subscribed
intervals = pd.cut(data['subscribed_days'], bins=[0, data['subscribed_days'].quantile(0.25), data['subscribed_days'].quantile(0.5), data['subscribed_days'].quantile(0.75)])

# Calculate the probability of churn 0 or churn 1 for each interval
churn_prob_by_interval = data.groupby(intervals)['churn'].value_counts(normalize=True).unstack()

print("Probability of Churn by Days Subscribed Interval:")
print(churn_prob_by_interval)




######################################### Mail Subscribed and Multi Screen  as the reasons ###############################################
# Calculate the percentage of each category for mail_subscribed
mail_subscribed_percentage = data['mail_subscribed'].value_counts(normalize=True) * 100

# Calculate the percentage of each category for multi_screen
multi_screen_percentage = data['multi_screen'].value_counts(normalize=True) * 100

print("Mail Subscribed:")
print(f"Yes: {mail_subscribed_percentage['yes']:.2f}%")
print(f"No: {mail_subscribed_percentage['no']:.2f}%")

print("Multi-Screen:")
print(f"Yes: {multi_screen_percentage['yes']:.2f}%")
print(f"No: {multi_screen_percentage['no']:.2f}%")

multi_screen_counts = data.groupby(['multi_screen', 'churn'])['multi_screen'].count().unstack()

print("Multi-Screen Counts:")
print(multi_screen_counts)

# Calculate the probability of churn 0 given multi_screen is yes
churn_prob_given_multi_screen_yes = data[data['multi_screen'] == 'yes']['churn'].value_counts(normalize=True)
churn_prob_given_multi_screen_no = data[data['multi_screen'] == 'no']['churn'].value_counts(normalize=True)
print("probability of churn to be 0 when multi screen is yes",churn_prob_given_multi_screen_yes[0])  # Churn 0 probability
print("probability of churn to be 1 when multi screen is yes",churn_prob_given_multi_screen_yes[1])
print("probability of churn to be 0 when multi screen is no",churn_prob_given_multi_screen_no[0])
print("probability of churn to be 1 when multi screen is no",churn_prob_given_multi_screen_yes[1])

mail_subscribed_counts = data.groupby(['mail_subscribed', 'churn'])['mail_subscribed'].count().unstack()

print("Mail_Subscribed Counts:")
print(mail_subscribed_counts)

# Calculate the probability of churn 0 given mail_subscribed is yes
churn_prob_given_mail_subscribed_yes = data[data['mail_subscribed'] == 'yes']['churn'].value_counts(normalize=True)
churn_prob_given_mail_subscribed_no = data[data['mail_subscribed'] == 'no']['churn'].value_counts(normalize=True)
print("probability of churn to be 0 when mail_subscribed is yes",churn_prob_given_mail_subscribed_yes[0])  # Churn 0 probability
print("probability of churn to be 1 when mail_subscribed is yes",churn_prob_given_mail_subscribed_yes[1])
print("probability of churn to be 0 when mail_subscribed is no",churn_prob_given_mail_subscribed_no[0])
print("probability of churn to be 1 when mail_subscribed is no",churn_prob_given_mail_subscribed_yes[1])

############################################# Weekly Mins as a reason ##############################################

weekly_mins_iqr = data['weekly_mins'].quantile(0.75) - data['weekly_mins'].quantile(0.25)

# Calculate the quartiles for days subscribed
q1 = data['weekly_mins'].quantile(0.25)
q2 = data['weekly_mins'].quantile(0.50)  # Median
q3 = data['weekly_mins'].quantile(0.75)

print("Weekly Mins:")
print(f"Q1: {q1:.2f}")
print(f"Q2 (Median): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"Weekly_mins_iqr : {weekly_mins_iqr:.2f}")

# Define the intervals for weekly mins
intervals = pd.cut(data['weekly_mins'], bins=[0, data['weekly_mins'].quantile(0.25), data['weekly_mins'].quantile(0.5), data['weekly_mins'].quantile(0.75)])

# Calculate the probability of churn 0 or churn 1 for each interval
churn_prob_by_interval = data.groupby(intervals)['churn'].value_counts(normalize=True).unstack()

print("Probability of Churn by weekly Mins Interval:")
print(churn_prob_by_interval)

################################################# Minimum mins daily as a reason ###########################################

min_mins_daily_iqr = data['min_mins_daily'].quantile(0.75) - data['min_mins_daily'].quantile(0.25)

# Calculate the quartiles for minimum mins daily
q1 = data['min_mins_daily'].quantile(0.25)
q2 = data['min_mins_daily'].quantile(0.50)  # Median
q3 = data['min_mins_daily'].quantile(0.75)

print("Minimum Mins Daily:")
print(f"Q1: {q1:.2f}")
print(f"Q2 (Median): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"Minimum_mins_dqily_iqr : {min_mins_daily_iqr:.2f}")

# Define the intervals for weekly mins
intervals = pd.cut(data['min_mins_daily'], bins=[0, data['min_mins_daily'].quantile(0.25), data['min_mins_daily'].quantile(0.5), data['min_mins_daily'].quantile(0.75)])

# Calculate the probability of churn 0 or churn 1 for each interval
churn_prob_by_interval = data.groupby(intervals)['churn'].value_counts(normalize=True).unstack()

print("Probability of Churn by Minimum Mins Daily Interval:")
print(churn_prob_by_interval)


################################################ Max Night mins as a Reason ###########################################################

max_night_mins_iqr = data['max_night_mins_weekly'].quantile(0.75) - data['max_night_mins_weekly'].quantile(0.25)
q1 = data['max_night_mins_weekly'].quantile(0.25)
q2 = data['max_night_mins_weekly'].quantile(0.50)  # Median
q3 = data['max_night_mins_weekly'].quantile(0.75)

print("Maximum night Mins :")
print(f"Q1: {q1:.2f}")
print(f"Q2 (Median): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"Maximum_night_weekly_iqr : {max_night_mins_iqr:.2f}")

# Define the intervals for max night mins
intervals = pd.cut(data['max_night_mins_weekly'], bins=[0, data['max_night_mins_weekly'].quantile(0.25), data['max_night_mins_weekly'].quantile(0.5), data['max_night_mins_weekly'].quantile(0.75)])

# Calculate the probability of churn 0 or churn 1 for each interval
churn_prob_by_interval = data.groupby(intervals)['churn'].value_counts(normalize=True).unstack()

print("Probability of Churn by max night Mins weekly Interval:")
print(churn_prob_by_interval)

###############################################  No. of videos watched as a reason#####################################

Number_of_videos_iqr = data['No.of_videos_watched'].quantile(0.75) - data['No.of_videos_watched'].quantile(0.25)
q1 = data['No.of_videos_watched'].quantile(0.25)
q2 = data['No.of_videos_watched'].quantile(0.50)  # Median
q3 = data['No.of_videos_watched'].quantile(0.75)

print("Number of videos watched :")
print(f"Q1: {q1:.2f}")
print(f"Q2 (Median): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"No.of_videos_watched_iqr : {Number_of_videos_iqr:.2f}")

# Define the intervals for max night mins
intervals = pd.cut(data['No.of_videos_watched'], bins=[0, data['No.of_videos_watched'].quantile(0.25), data['No.of_videos_watched'].quantile(0.5), data['No.of_videos_watched'].quantile(0.75)])

# Calculate the probability of churn 0 or churn 1 for each interval
churn_prob_by_interval = data.groupby(intervals)['churn'].value_counts(normalize=True).unstack()

print("Probability of Churn by Number of videos watched Interval:")
print(churn_prob_by_interval)


###############################################  Max inactive days  as a reason#####################################

max_inactive_days_iqr = data['max_inactive_days'].quantile(0.75) - data['max_inactive_days'].quantile(0.25)
q1 = data['max_inactive_days'].quantile(0.25)
q2 = data['max_inactive_days'].quantile(0.50)  # Median
q3 = data['max_inactive_days'].quantile(0.75)

print("Maximum Inactive days :")
print(f"Q1: {q1:.2f}")
print(f"Q2 (Median): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"max_inactive_days_iqr : {max_inactive_days_iqr:.2f}")

# Define the intervals for max night mins
intervals = pd.cut(data['max_inactive_days'], bins=[0, 2, data['max_inactive_days'].quantile(0.5), data['max_inactive_days'].quantile(0.75)])

# Calculate the probability of churn 0 or churn 1 for each interval
churn_prob_by_interval = data.groupby(intervals)['churn'].value_counts(normalize=True).unstack()

print("Probability of Churn by maximum inactive days to the Interval:")
print(churn_prob_by_interval)


###############################################  No. of customer care calls  as a reason#####################################

calls_customer_care_iqr = data['calls_customer_care'].quantile(0.75) - data['calls_customer_care'].quantile(0.25)
q1 = data['calls_customer_care'].quantile(0.25)
q2 = data['calls_customer_care'].quantile(0.50)  # Median
q3 = data['calls_customer_care'].quantile(0.75)

print("Number of calls to the customer care watched :")
print(f"Q1: {q1:.2f}")
print(f"Q2 (Median): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"calls_customer_care_iqr : {calls_customer_care_iqr:.2f}")

# Define the intervals for max night mins
intervals = pd.cut(data['calls_customer_care'], bins=[0, data['calls_customer_care'].quantile(0.5), data['calls_customer_care'].quantile(0.75)])

# Calculate the probability of churn 0 or churn 1 for each interval
churn_prob_by_interval = data.groupby(intervals)['churn'].value_counts(normalize=True).unstack()

print("Probability of Churn by Number of calls to the Interval:")
print(churn_prob_by_interval)