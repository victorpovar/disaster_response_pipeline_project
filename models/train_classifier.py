# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load the data from the sqlite db.
    
    Keyword arguments:
    database_filepath -- filepath to the sqlite db
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * from ETL_Pipeline', con = engine)
    X = df.message
    Y = df [df.columns[3:]]
    
    return X, Y, Y.columns


def tokenize(text):
    """
    Normalized the case, remove punctuations, tokenize the text and remove stop words
    
    Keyword arguments:
    text -- text that should be tokenized
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """
    Create the pipeline with gridsearch to be used for model development.
    
    Keyword arguments:
    text -- text that should be tokenized
    """

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('multiple_output_classifier', MultiOutputClassifier (RandomForestClassifier()))
    ])
    
    parameters = {"multiple_output_classifier__estimator__n_estimators" : [1, 100]}

    cv = GridSearchCV (estimator = pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model.
    
    Keyword arguments:
    model -- model that should be evaluated
    X_test -- X_test data
    Y_test -- Y_test data
    category_names -- category names
    """
    
    # predict on test data
    y_pred = model.predict(X_test)
    
    for i in range (1, Y_test.shape[0]):
        print (classification_report(Y_test.values[i], y_pred[i]))

def save_model(model, model_filepath):
    """
    Save the model.
    
    Keyword arguments:
    model -- model that should be saved
    model_filepath -- file path that where the model should be saved
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
