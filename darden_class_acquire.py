import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire Titanic Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
def new_titanic_data():
    '''
    This function reads the titanic data from the Codeup db
    into a df and returns the df.
    '''
    sql_query = 'SELECT * FROM passengers'
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    df.to_csv('titanic_df.csv')
    return df

def get_titanic_data(cached=False):
    '''
    This function reads in titanic data from Codeup database if cached == False 
    or if cached == True reads in titanic df from a csv file, returns df
    '''
    if cached or os.path.isfile('titanic_df.csv') == False:
        df = new_titanic_data()
    else:
        df = pd.read_csv('titanic_df.csv', index_col=0)
    return df

###################### Acquire Iris Data ######################

