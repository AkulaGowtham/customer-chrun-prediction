from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# Load data into a Pandas dataframe
df = pd.read_csv('subcriptions.csv')

print("shape of data set before pre-processing",df.shape)
#print(df.isna().sum())
df.dropna(inplace=True)
df=df.drop('contact_no',axis=1)
print("shape of data set after pre-processing",df.shape)
#label encoding

print("The unique values of sex before lebel-encoding",df['sex'].unique())
print("The unique values of multi_screen before lebel-encoding",df['multi_screen'].unique())
print("The unique values of mail_subscribed before lebel-encoding",df['mail_subscribed'].unique())

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df['sex'] = LE.fit_transform(df.sex)
df['multi_screen'] = LE.fit_transform(df.multi_screen)
df['mail_subscribed'] = LE.fit_transform(df.mail_subscribed)

print("The unique values of sex after lebel-encoding",df['sex'].unique())
print("The unique values of multi_screen after lebel-encoding",df['multi_screen'].unique())
print("The unique values of mail_subscribed after lebel-encoding",df['mail_subscribed'].unique())
#seperate into dependent and independent variables
x = df.iloc[:,:-1]
y = df.iloc[:, -1]

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=0)

# Instantiate the logistic regression model with default hyperparameters
model = RandomForestClassifier(n_estimators=100)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the performance of the model
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)*100))
print('Precision: {:.3f}'.format(precision_score(y_test, y_pred)*100))
print('Recall: {:.3f}'.format(recall_score(y_test, y_pred)*100))
print('F1-score: {:.3f}'.format(f1_score(y_test, y_pred)*100))
print('ROC AUC: {:.3f}'.format(roc_auc_score(y_test, y_pred)*100))
'''
from sklearn.model_selection import cross_val_score


# Perform cross-validation
scores = cross_val_score(model,x, y, cv=5, scoring='r2')

# Print the cross-validation scores
print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())
print("Standard Deviation of CV Scores:", scores.std())'''

'''import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()'''
'''
#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logreg.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("Accuracy when logistic regression:", accuracy*100)
print("Precision when logistic regression used:", precision*100)
print("Recall when logistic regression used:", recall*100)
print("F1 Score when logistic regression used:", f1*100)
print("ROC AUC Score when logistic regression used:", roc_auc*100)
# SVM
from sklearn.svm import SVC
# Create an SVM classifier object
svm_classifier = SVC(kernel='linear')

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
# Print the accuracy
print("Accuracy when the SVM is used:", accuracy)
print("Precision when svm used:", precision*100)
print("Recall when svm used:", recall*100)
print("F1 Score when svm used:", f1*100)
print("ROC AUC Score when svm used:", roc_auc*100)'''

'''
# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier
# Create the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()

# Train the model
gb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy*100)
print("Precision:", precision*100)
print("Recall:", recall*100)
print("F1 Score:", f1*100)
print("ROC AUC Score:", roc_auc*100)'''

#dumping the model
import pickle
pickle.dump(model, open('subscribe_RF_classifier.pkl', 'wb'))
subscribe_RF_classifier = pickle.load(open('subscribe_RF_classifier.pkl', 'rb'))