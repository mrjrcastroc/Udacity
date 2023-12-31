import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    This function load data from csv files and merge to a single pandas dataframe
    
    Inputs:

    messages_filepath: filepath to messages csv file
    categories_filepath: filepath to categories csv file
    

    Returns:

    df: dataframe merging categories and messages

    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='inner', left_on = 'id', right_on ='id')
    return df 

def clean_data(df):
    '''
    clean_data
    This function splits the categories data and after transform each category into a binary column
    Inputs:

    df: merged dataframe 

    Returns:

    df: dataframe with all categories transformed into binary columns

    '''    
    # create a dataframe of the 36 individual category columns
    categories =  df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.head(1).values[0]
    # extract a list of new column names for categories.
    category_colnames = pd.Series(row).apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    # drop nulls
    df = df.dropna()
    # removing the values with classification 2 in the column related
    df = df.query('related != 2')
    return df

def save_data(df, database_filename):
    '''
    save_data
    This function takes a dataframe and saves the data into a sqlite database
    Inputs:

    df:  dataframe 
    database_filename: database name

    Returns:

    creates the table tb_messages in the sqlite database 

    '''      
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('tb_messages', engine, index=False,if_exists='replace')  

def main():
    print('ok')
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