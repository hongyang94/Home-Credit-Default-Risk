import pandas as pd
import numpy as np


def one_hot_encode(df):
    original_columns = list(df.columns)
    categoricals = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categoricals, dummy_na=False)
    categoricals = [c for c in df.columns if c not in original_columns]
    return df, categoricals


def aggregate(df, aggregates):
    df = df.groupby('SK_ID_CURR').agg(aggregates)
    df.columns = pd.Index(['{}{}_{}'.format('PREV_', e[0], e[1].upper()) for e in df.columns.tolist()])
    return df.reset_index()


def process_previous_application():

    df = pd.read_csv('../../data/previous_application.csv')
    df, categorical = one_hot_encode(df)

    # Basic cleaning
    df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Feature engineering: ratios and difference
    df['APPLICATION_CREDIT_DIFF'] = df['AMT_APPLICATION'] - df['AMT_CREDIT']
    df['APPLICATION_CREDIT_RATIO'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT']/df['AMT_ANNUITY']
    df['DOWN_PAYMENT_TO_CREDIT'] = df['AMT_DOWN_PAYMENT'] / df['AMT_CREDIT']

    categorical_agg = {key: ['mean'] for key in categorical}
    aggregates = {
        'SK_ID_PREV': ['nunique'],
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'RATE_INTEREST_PRIMARY': ['min', 'max', 'mean'],
        'RATE_INTEREST_PRIVILEGED': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'SELLERPLACE_AREA': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['max', 'mean'],
        'DAYS_FIRST_DRAWING': ['min', 'max', 'mean'],
        'DAYS_FIRST_DUE': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE': ['min', 'max', 'mean'],
        'DAYS_TERMINATION': ['max'],
        'NFLAG_INSURED_ON_APPROVAL': ['sum', 'mean'],
        'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean'],
        'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
        'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
        'DOWN_PAYMENT_TO_CREDIT': ['mean'],
        **categorical_agg
    }

    agg_prev = aggregate(df, aggregates)
    agg_prev.to_csv('../../processed data/prev_app_processed.csv', index=False)


if __name__ == "__main__":
    process_previous_application()
