import pandas as pd
import os
from sklearn.model_selection import train_test_split

NUM_FEATURES1 = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE',
                 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                 'EXT_SOURCE_2', 'DAYS_LAST_PHONE_CHANGE', 'OBS_30_CNT_SOCIAL_CIRCLE',
                 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                 'document', 'address']  # no missing indicator
NUM_FEATURES2 = ['EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']  # add missing indicator
CAT_FEATURES = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START',
                'HOUR_APPR_PROCESS_START', 'ORGANIZATION_TYPE']
# binary_features = ['FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL',
#                    'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',
#                    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','FLAG_DOCUMENT_2',
#                    'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
#                    'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
#                    'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
#                    'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
BINARY_FEATURES = ['FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                   'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']


def preprocess_data():
    path = 'data/application_data.csv'
    df = pd.read_csv(path)

    # feature engineering on address discrepancies and document count
    df['document'] = df['FLAG_DOCUMENT_2']+df['FLAG_DOCUMENT_3']+df['FLAG_DOCUMENT_4']+df['FLAG_DOCUMENT_5']+df['FLAG_DOCUMENT_6']\
        + df['FLAG_DOCUMENT_7']+df['FLAG_DOCUMENT_8']+df['FLAG_DOCUMENT_9']+df['FLAG_DOCUMENT_10']+df['FLAG_DOCUMENT_11']\
        + df['FLAG_DOCUMENT_12']+df['FLAG_DOCUMENT_13']+df['FLAG_DOCUMENT_14']+df['FLAG_DOCUMENT_15']+df['FLAG_DOCUMENT_16']\
        + df['FLAG_DOCUMENT_17']+df['FLAG_DOCUMENT_18'] + \
        df['FLAG_DOCUMENT_19']+df['FLAG_DOCUMENT_20']+df['FLAG_DOCUMENT_21']

    df['address'] = df['REG_REGION_NOT_LIVE_REGION']+df['REG_REGION_NOT_WORK_REGION']+df['LIVE_REGION_NOT_WORK_REGION']\
        + df['REG_CITY_NOT_LIVE_CITY'] + \
        df['REG_CITY_NOT_WORK_CITY']+df['LIVE_CITY_NOT_WORK_CITY']

    drop_col = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
                'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

    df.drop(columns=drop_col, inplace=True)

    # drop columns with over 50% missing and the ID column, FLAG_MOBIL is 100% 1, REGION_RATING_CLIENT_W_CITY correlates with REGION_RATING_CLIENT
    drop_col = ['OWN_CAR_AGE', 'EXT_SOURCE_1', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
                'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
                'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE',
                'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
                'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
                'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
                'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
                'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
                'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'SK_ID_CURR', 'FLAG_MOBIL', 'REGION_RATING_CLIENT_W_CITY']

    df.drop(columns=drop_col, inplace=True)

    # preprocessing outside of pipeline
    # df['DAYS_LAST_PHONE_CHANGE']=df['DAYS_LAST_PHONE_CHANGE'].fillna(1)
    # df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].fillna(df['CNT_CHILDREN']+1)
    # df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna("Others")
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: 0})
    # df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].fillna("Missing")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(df[NUM_FEATURES1+NUM_FEATURES2+CAT_FEATURES+BINARY_FEATURES], df['TARGET'],
                                                        stratify=df['TARGET'], test_size=0.2, random_state=42)

    # Save data
    split_destination_folder = './data/processed'
    if not os.path.exists(split_destination_folder):
        os.makedirs(split_destination_folder)

    X_train.to_csv('./data/processed/X_train.csv', index=False)
    X_test.to_csv('./data/processed/X_test.csv', index=False)
    y_train.to_csv('./data/processed/y_train.csv', index=False)
    y_test.to_csv('./data/processed/y_test.csv', index=False)

    file_locations = {
        'X_train_dir': './data/processed/X_train.csv',
        'X_test_dir': './data/processed/X_test.csv',
        'y_train_dir': './data/processed/y_train.csv',
        'y_test_dir': './data/processed/y_test.csv',
    }

    return file_locations
