"""
Data Preprocessing
Project: Disaster Response Pipeline
Sample Script Syntax:
> python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>
Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db
Input Description:
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response.db)
"""
# Import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """
    Load data function

    Input:
        messages_filepath -> file path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined data containing messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df


def clean_data(df):

    """
    Clean category data function

    Input:
        df -> Combined data frame containing messages and categories
    Outputs:
        df -> Combined data frame containing messages and cleaned categories
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';',expand=True))
    row = categories.iloc[0]
    category_colnames = [col.split('-')[0] for col in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    # drop the rows with all category response nan
    df.dropna(axis=0,how='all', subset= category_colnames, inplace=True)
    # drop the child_alone column with all category response 0
    df.drop(['child_alone'], axis=1, inplace=True)
    # set the invalid response 2 in the 'related' category to 0
    df.loc[df.index[df['related']==2], 'related']=0

    return df


def save_data(df, database_filename):

    """
    Save data function (to SQLite Database)

    Arguments:
        df -> Combined data containing messages and cleaned categories
        database_filename: database name
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


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
