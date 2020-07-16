import pandas as pd


def one_hot_encode(df):
    categoricals = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categoricals, dummy_na=True)
    return df


def aggregate(df, aggregates):
    df = df.groupby('SK_ID_CURR').agg(aggregates)
    df.columns = pd.Index(['{}{}_{}'.format('BUREAU_', e[0], e[1].upper()) for e in df.columns.tolist()])
    return df.reset_index()


def process_bureau():

    bureau = pd.read_csv('../../data/bureau.csv')

    # One-hot encode
    bureau = one_hot_encode(bureau)
    bureau_balance = pd.read_csv('../../processed data/bureau_balance_processed.csv')

    # Join bureau balance features
    bureau = bureau.merge(bureau_balance, how='left', on='SK_ID_BUREAU')

    # sum of all status (days past due)
    bureau['STATUS_12345'] = 0
    for i in range(1,6):
        bureau['STATUS_12345'] += bureau['STATUS_{}'.format(i)]

    aggregates = {
        'SK_ID_BUREAU': ['nunique'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean', 'sum'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'sum'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['min', 'max', 'sum', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'sum', 'mean'],
        'AMT_ANNUITY': ['mean'],
        'MONTHS_BALANCE_MIN': ['mean', 'sum'],
        'MONTHS_BALANCE_MAX': ['mean', 'sum'],
        'MONTHS_BALANCE_MEAN': ['mean', 'var'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        # Categorical
        'STATUS_0': ['mean'],
        'STATUS_1': ['mean'],
        'STATUS_2': ['mean'],
        'STATUS_3': ['mean'],
        'STATUS_4': ['mean'],
        'STATUS_5': ['mean'],
        'STATUS_12345': ['mean'],
        'STATUS_C': ['mean'],
        'STATUS_X': ['mean'],
        'CREDIT_ACTIVE_Active': ['mean'],
        'CREDIT_ACTIVE_Closed': ['mean'],
        'CREDIT_ACTIVE_Sold': ['mean'],
        'CREDIT_TYPE_Consumer credit': ['mean'],
        'CREDIT_TYPE_Credit card': ['mean'],
        'CREDIT_TYPE_Car loan': ['mean'],
        'CREDIT_TYPE_Mortgage': ['mean'],
        'CREDIT_TYPE_Microloan': ['mean']
    }

    agg_bureau = aggregate(bureau, aggregates)
    agg_bureau.to_csv('../../processed data/bureau_processed.csv', index=False)


if __name__ == "__main__":
    process_bureau()
