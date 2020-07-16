import pandas as pd
import numpy as np


def label_encoder(df):

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df


def process_train_test():

    train = pd.read_csv('../../data/application_train.csv')
    test = pd.read_csv('../../data/application_test.csv')
    df = train.append(test)

    # Data cleaning
    df = df[df['CODE_GENDER'] != 'XNA']  # 4 people with XNA code gender
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Flag_document features - sum
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)

    # New features based on External sources
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    for function_name in ['min', 'max', 'mean', 'var']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    # feature engineering
    df['CREDIT_TO_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['EMPLOYED_TO_INCOME'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['BIRTH_TO_INCOME'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']

    df['EMPLOYED_TO_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

    df = label_encoder(df)
    df.to_csv('../../processed data/train_test_processed.csv', index=False)

    return df


if __name__ == "__main__":
    process_train_test()
