import sqlite3
import pandas as pd
import numpy as np


def extract_ground_truth_features(sqlite_file):
    cnx = sqlite3.connect(sqlite_file)
    df = pd.read_sql_query("""SELECT 
            FOD_ID, FIRE_NAME, datetime(DISCOVERY_DATE) as DISCOVERY_DATE, 
            STAT_CAUSE_DESCR, datetime(CONT_DATE) as CONT_DATE, FIRE_SIZE, 
            FIRE_SIZE_CLASS, LATITUDE, LONGITUDE, STATE 
            FROM fires""", cnx)
    df.to_csv('./data/wildfire_occurences.csv')


if __name__ == "__main__":
    df = extract_ground_truth_features('./data/FPA_FOD_20170508.sqlite')
