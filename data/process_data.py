import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories data and merge them into a single dataframe.
    
    Keyword arguments:
    messages_filepath -- filepaths of the messages file
    categories_filepath -- filepaths of the categories file
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath, index_col=0)
    # load categories dataset
    categories = pd.read_csv(categories_filepath, index_col=0)
    
    # merge datasets
    df = pd.concat([messages, categories], sort = True, axis=1)

    return df
    
def clean_data(df):
    """
    Clean the provided dataframe by splitting the categories column into 36 columns and clean each of the categories rows to remove prefixes. Remove duplicates.
    
    Keyword arguments:
    df -- the dataframe that should be cleaned
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = []
    for r in row:
        category_colnames.append (r[:r.find("-")])
        
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] =  categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort = True, axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save the provided dataframe to the specified SQL database. Drop the table if it already exists.
    
    Keyword arguments:
    df -- the dataframe that should be saved
    database_filename -- database filename
    """
    engine = create_engine('sqlite:///' + database_filename)
    # drop the test table in case it already exists
    engine.execute("DROP TABLE IF EXISTS ETL_Pipeline")
    # push the data from df to SQL
    df.to_sql('ETL_Pipeline', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
