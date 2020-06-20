# Disaster Response Pipeline Project

This project is focused on the analysis of disaster data from Figure Eight. 

The project consists of a natural language classifer that classifies messages as belonging to various disaster relief agencies.

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
 
# Data
The data for the analysis was obtained from the Figure Eight: https://appen.com/

# Project Components

There are three components in this project.
1. ETL Pipeline - a Python script, process_data.py, a data cleaning pipeline:

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

2. ML Pipeline - a Python script, train_classifier.py, a machine learning pipeline:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file

3. Flask Web App:

    Classifies the user messages


# Acknowledgement
This analysis was performed as part of the Data Science Udacity Nanodegree
