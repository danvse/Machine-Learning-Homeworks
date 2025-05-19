#data imports
import pandas as pd #load in and create a dataframe
import numpy as np #takes a list of data to convert into 2D or 3D arr 

#machine learning imports
from sklearn.model_selection import train_test_split, GridSearchCV #creat a training and testing set, and test multiple scenarios for best performance
from sklearn.preprocessing import MinMaxScaler #creating data for the model, convert items to 0 and 1
from sklearn.neighbors import KNeighborsClassifier #the model we will be using
from sklearn.metrics import accuracy_score, confusion_matrix
#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("titanic.csv") #loads in our data
data.info() #get info of datatype
print(data.isnull().sum())#finds how many missing values we have for cleaning

#data cleaning
def preprocess_data(df): #takes a dataframe
    df.drop(columns=["PassengerId","Name","Ticket", "Cabin"], inplace=True)# removes columns passengerid,name,ticket, and cabin because irrevelent to our data

    df["Embarked"].fillna("S", inplace=True)#targets and fills the missing data in Embarked with "S"
    df.drop(columns=["Embarked"], inplace=True)#drop embarked column
    fill_missing_ages(df) #calls our function missing age
    #convert gender
    df["Sex"] = df["Sex"].map({'male':1, "female":0})#maps the genders into numbers, make = 1 and females = 0
    
    #Feature Engineering, creates new column in our data
    df["FamilySize"] = df["SibSp"] + df["Parch"] #combines Sib and Parch columns into one comlumn named familysize
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)# if family size is 0, insert 1. otherwise insert 0
    df["FareBin"] = pd.qcut(df["Fare"],4,labels=False) #tries to create a range for finding out cost of tickets, creates 4 catagory from Rares
    df ["AgeBin"] = pd.cut(df["Age"], bins= [0,12,20,40,60, np.inf], labels=False)#create multiple bins to account for high age ranges in integer format

    return df #returns our dataframe

def fill_missing_ages(df): #gets average age of the class, Ex: average of people in age group 12 etc
    age_fill_map = {} #creates a dictionary to store passenger class
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map: #
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()#take our dataframe, targets pclass column, if pclass target is what the pclass we are looking for, get the age and median value

    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)# apply custom function lambda, checks and applys that function if age is missing, ecks if any of the ages are missing

    #Machine Learning side
data = preprocess_data(data)#calls and update the data

#Features
X = data.drop(columns = ["Survived"])#gets the answer to who survived and who did not
y = data["Survived"] #answer to check the model

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42) #flashcards for testing and training , sees front and back of flash cards when in learning and then we test to see how well it learned. Learns with 75% of data and 25% used for testing the model

#ML preprocessing
scaler = MinMaxScaler() #Scales data
X_train = scaler.fit_transform(X_train)#front of flashcard
X_test = scaler.transform(X_test)#the other front of flashcard

#get best model possible
#KNN model, collect data from nearest datapoints, simulation runs multiple times
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors":range(1,21),# checks from 1 to 21 points
        "metric" : ["euclidean", "manhattan", "minkowski"], #different ways to check data built in the machien learning model
        "weights" : ["uniform","distance"]
        }
    model = KNeighborsClassifier() #model we imported, performs grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs =-1) #giving grid search parameters
    grid_search.fit(X_train, y_train) #trains the data giving them the front and back of flashcards which we called x and y 
    return grid_search.best_estimator_
best_model = tune_model(X_train, y_train)#gives whatever model works the best that was determined by tune_model

def evaluate_model(model, X_test, y_test):#how well the model performed
    predictions = model.predict(X_test)#checks how well the model performed from predictions
    accuracy = accuracy_score(y_test, predictions)#answers to our "test", the other 25% flashcards
    matrix = confusion_matrix(y_test, predictions)
    return accuracy, matrix

accuracy,matrix = evaluate_model(best_model, X_test, y_test)
#visual representation of our tests
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)

#model ploting
def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt="d",xticklabels=["Survived", "Not Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")#title of graph
    plt.xlabel("Predicted Value")#labels
    plt.show()
plot_model(matrix)