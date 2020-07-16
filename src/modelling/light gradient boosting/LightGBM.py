import gc
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_feature_importance(df):

    cols = df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = df.loc[df.feature.isin(cols)]
    plt.figure(figsize=(25, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.tight_layout
    plt.savefig('../../output/feature importance/lgbm_importances.png')


def lgbm(df, num_bags):

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print('Train: ' + str(train_df.shape) + ', Test: ' + str(test_df.shape))

    # Create arrays to store results
    test_df_prediction = np.zeros(test_df.shape[0])

    # get all features
    features = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'index']]

    for i in range(num_bags):

        print('Bagging round ' + str(i + 1))
        train_index = train_df.sample(frac=0.8, replace=False).index
        test_index = [index for index in train_df.index if index not in train_index]

        split_train = lgb.Dataset(data=train_df[features].loc[train_index], label=train_df['TARGET'].loc[train_index], free_raw_data=False, silent=True)
        split_test = lgb.Dataset(data=train_df[features].loc[test_index], label=train_df['TARGET'].loc[test_index], free_raw_data=False, silent=True)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            "learning_rate": 0.02, #0.016179103202121937,
            "num_leaves": 20,
            "colsample_bytree": 0.8187763990353812,
            "subsample": 0.9867432001602304,
            "max_depth": 8,
            "reg_alpha": 0.048447515906389196,
            "reg_lambda": 0.07755349459792096,
            "min_split_gain": 0.02418255863819766,
            "min_child_weight": 60, #39.90789678572602,
            'nthread': 4,
            'subsample_freq': 1,
            'seed': 0,
            'verbose': -1,
            'metric': 'auc',
        }

        print('training...')
        clf = lgb.train(
            params=params,
            train_set=split_train,
            num_boost_round=10000,
            valid_sets=[split_train, split_test],
            early_stopping_rounds=200,
            verbose_eval=False
        )

        print('predicting')
        test_df_prediction += clf.predict(test_df[features]) / num_bags

        feature_importance_df = pd.DataFrame()
        feature_importance_df["feature"] = features
        feature_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        get_feature_importance(feature_importance_df)

    # create submission file
    submit = test_df[['SK_ID_CURR']].copy()
    submit['TARGET'] = test_df_prediction
    submit[['SK_ID_CURR', 'TARGET']].to_csv('../../output/submit/lgbm_submission.csv', index=False)


if __name__ == "__main__":

    # set to ignore warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Read
    df = pd.read_csv('../../processed data/train_test_processed.csv')

    # bureau and bureau balance
    bureau_balance = pd.read_csv('../../processed data/bureau_processed.csv')
    df = pd.merge(df, bureau_balance, how='left', on='SK_ID_CURR')

    # previous application
    previous_application = pd.read_csv('../../processed data/prev_processed.csv')
    df = pd.merge(df, previous_application, how='left', on='SK_ID_CURR')

    # installment payments
    installments = pd.read_csv('../../processed data/installments_processed.csv')
    df = pd.merge(df, installments, how='left', on='SK_ID_CURR')

    # credit card balance
    credit_card = pd.read_csv('../../processed data/credit_processed.csv')
    df = pd.merge(df, credit_card, how='left', on='SK_ID_CURR')

    # pos cash balance
    pos_cash = pd.read_csv('../../processed data/pos_cash_processed.csv')
    df = pd.merge(df, pos_cash, how='left', on='SK_ID_CURR')

    # train and predict
    lgbm(df, num_bags=5)
