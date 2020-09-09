import pandas as pd
import numpy as np

from darden_class_acquire import get_titanic_data, get_iris_data

###################### Prep Iris Data ######################

def prep_iris(cached=True):
    '''
    This function acquires and prepares the iris data from a local csv, default.
    Passing cached=False acquires fresh data from Codeup db and writes to csv.
    Returns the iris df with dummy variables encoding species.
    '''
    df = get_iris_data(cached)
    df = df.drop(columns='species_id').rename(columns={'species_name': 'species'})
    species_dummies = pd.get_dummies(df.species, drop_first=True)
    df = pd.concat([df, species_dummies], axis=1)
    return df

###################### Prep Titanic Data ######################

