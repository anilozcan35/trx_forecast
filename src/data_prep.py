import os
import pickle
from darts import TimeSeries
from darts.models.forecasting import lgbm
from darts.models.forecasting.lgbm import LightGBMModel
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

with open("C:\\Users\\aozcan\\PycharmProjects\\trx_forecast\\data\\ts", "rb") as file:
    ts = pickle.load(file)


def lgbm_contructor(params_lgbm, train, test):
    """
    :param params_lgbm: Lightgbm Params passed from optuna trial
    :param train: train data
    :param test:  test-val data
    :return: mean squared error metrik
    """
    lgbm = LightGBMModel(
        verbose=-1,
        categorical_static_covariates=["shop", "item", "item_category_id", "cat_cluster"],
        use_static_covariates=True, **params_lgbm, n_jobs=-1
        # static covariates veri içerisindeki id'ler aslında
    )
    lgbm.fit(series=[*train])
    preds = lgbm.predict(1, series=train)
    preds_pd = pd.DataFrame()
    val_pd = pd.DataFrame()
    for i in tqdm(range(len(preds))):
        preds_pd = preds_pd.append(preds[i].pd_dataframe())
        test_pd = val_pd.append(test[i].pd_dataframe())
    mse = mean_squared_error(y_true=test_pd.amount.tolist(), y_pred=preds_pd.amount.tolist())
    return mse


def objective(trial, train, test):
    """
    :param test: test data
    :param train: train data
    :param trial: optuna trial parameters
    :return: mse value returned from Lightgbm Constructor
    """
    num_leaves_list = [int(2 ** x) for x in np.arange(4, 8, 1)]
    lags_list = [[-1, -2, -3, -4, -5, -6, -12, -24]]
    params_lgbm = {
        'n_estimators': trial.suggest_int("n_estimators", 3000, 5100, step=100),
        'num_leaves': trial.suggest_categorical("num_leaves", num_leaves_list),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10, step=1),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.201, step=0.01),
        'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 10.0, step=1),
        'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 10.0, step=1),
        'lags': trial.suggest_categorical("lags", lags_list)
    }
    return lgbm_contructor(params_lgbm, train, test)


train_ts = []
test_ts = []

for series in tqdm(ts):
    train_series, val_series = series[:-1], series[-1]
    train_ts.append(train_series)
    test_ts.append(val_series)

with open("./data/train", "wb") as file:
    pickle.dump(train_ts, file)

with open("./data/test", "wb") as file:
    pickle.dump(test_ts, file)

func = lambda trial: objective(trial, train_ts, test_ts)  #This is for to pass args into objective function
study = optuna.create_study(directions=["minimize"])
study.optimize(func, n_trials=1)  #func değişkeni optimize edilmek üzere yollanıyor.

with open("./data/train", "rb") as file:
    train_ts = pickle.load(file)

with open("./data/test", "rb") as file:
    test_ts = pickle.load(file)

def fit_Lightgbm(train):
    """
    :param train: train seti
    :return: fit edilmiş Lightgbm objesi
    """
    print("######### fit_lightgbm ############")
    lgbm = LightGBMModel(verbose=-1,
                         categorical_static_covariates=["shop", "item", "item_category_id", "cat_cluster"],
                         use_static_covariates=True,
                         n_jobs=-1,
                         lags=[-1, -2, -3, -6, -9, -12, -24],
                         output_chunk_length=1
                         )
    lgbm.fit(series=[*train])
    return lgbm


def predict_Lightgbm(lightgbm, train):
    """
    :param lightgbm: lightgbm objesi
    :param train: train seti
    :return: Darts time series objelerinden oluşan liste tipi predictionlar
    """
    print("########### prediction #############")
    preds = lightgbm.predict(1, series=[*train])
    return preds


def get_pred_dataframe(preds):
    """
    predictionları pandas dataframe'i olarak transform eder
    :param preds: Darts timeseries objeclerinden oluşan liste
    :return: prediction dataframe'i
    """
    print("####### prediction pandas ########")
    #Boş bir dataframe'e append edilecek.
    preds_list = []
    for i in tqdm(range(len(preds))):
        #Predictionlardan pandas dataframe'i elde ediliyor.
        preds_partial = pd.DataFrame(columns=["shop", "item", "item_category_id", "cat_cluster", "pred"])

        preds_partial.loc[i] = [preds[i].static_covariates["shop"].values[0],
                                preds[i].static_covariates["item"].values[0],
                                preds[i].static_covariates["item_category_id"].values[0],
                                preds[i].static_covariates["cat_cluster"].values[0],
                                round(preds[i].pd_dataframe()["amount"].values[0], 0)
                                ]
        #Ana dataframe'e ekleniyor.
        preds_list.append(preds_partial)

    results = pd.concat(preds_list)

    return results


def pipeline_lgbm(train):
    """
    tüm fonksiyonların çağrıldığı pipeline fonksiyonu

    :param train: train seti
    :return: predictionlara ait pandas dataframe'i döndürür
    """
    lgbm = fit_Lightgbm(train)
    preds = predict_Lightgbm(lgbm, train)
    results = get_pred_dataframe(preds)

    return results


results = pipeline_lgbm(train=train_ts)


results[results["pred"] > 100]

results[(results["shop"] == 154) & (results["pred"] > 10)]