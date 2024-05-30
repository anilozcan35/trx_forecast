from datetime import datetime
import pandas as pd
from darts import TimeSeries
from darts.utils import *
from darts.utils import missing_values
from tqdm import tqdm
import os
import importlib
import pickle
from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode
from joblib import Parallel, delayed
import multiprocessing
import utils

importlib.reload(utils)


# Bu scriptte ben son 1 yılda ve toplam veri setinin uzunluğu 0.5'ten kısa olanları eledim.
# ancak yeni bir ürün satışa sunulduğunda burada satış olmasına rağmen forecast yapılmaması gibi bir durum olacak.
# Bunun için son bir yıldaki satışlarına göre eleme yapıldıktan sonra, 0.5 koşulu ile sparse seriesler elenmezse,
# bir ürün satışa girdikten sonra modele sokulabilir.
# Mesela bir ürün son yılda 0 satıldıysa sıfır basarız, Eğer satışı varsa smoothing ile modellenebilir.

def define_paths():
    CURRENT_DIR = os.getcwd()
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    DATA_PATH = os.path.join(BASE_DIR, 'data')  # pathler kontrol edilmeli.
    CAT_PATH = os.path.join(DATA_PATH, "category_list.csv")
    ITEM_PATH = os.path.join(DATA_PATH, "item_list.csv")
    SHOP_PATH = os.path.join(DATA_PATH, "shop_list.csv")
    TRX_PATH = os.path.join(DATA_PATH, "transaction.csv")
    return ITEM_PATH, CAT_PATH, SHOP_PATH, TRX_PATH


def get_dataframes():
    ITEM_PATH, CAT_PATH, SHOP_PATH, TRX_PATH = define_paths()

    item_df = pd.read_csv(ITEM_PATH, index_col=0)
    cat_df = pd.read_csv(CAT_PATH, index_col=0)
    shop_df = pd.read_csv(SHOP_PATH, index_col=0)
    trx_df = pd.read_csv(TRX_PATH, index_col=0)

    return item_df, cat_df, shop_df, trx_df


def data_preprocess(trx_df):
    # son 1 yıldır satmayan ürünleri 0 olarak forecast edicez.
    # Data boyutu date-store-item yapıldığında çok büyüyor.
    # Böylece bu büyümenin de önüne geçmiş olacağız.

    trx_df["date"] = pd.to_datetime(trx_df["date"], format="%d.%m.%Y")  # date convert
    max_date = trx_df["date"].max()  # max_date
    max_date_shopitem = trx_df.groupby(["shop", "item"])["date"].max().reset_index()  # shop-item max date
    zero_preds = max_date_shopitem[max_date_shopitem["date"] < "2014-09-30"]

    # 0 pred basacaklarımızı kararlaştırdık. Şimdi kalanlar üzerinde manipulasyon yapabiliriz.
    zero_preds = zero_preds.drop(columns=["date"], axis=1).drop_duplicates()
    merged = trx_df.merge(zero_preds, how="outer", on=["shop", "item"],
                          indicator=True)  # iki dataframe mergelenip eşlenmeyenler alınacak.
    unmatched = merged[merged["_merge"] != "both"]
    unmatched.drop(columns=["_merge"], inplace=True, axis=1)
    # control = unmatched.groupby(["shop", "item"])["date"].max().reset_index() # shop-item max date

    # elimizdeki store-item ikilileri biraz daha olsa azaldı artık.
    unmatched.groupby(["shop", "item"]).count()
    unmatched.set_index("date", inplace=True)
    grouper = unmatched.groupby([pd.Grouper(freq='ME'), 'shop', "item"])  # item bazında grupla- aylık dataframe'e dön
    trx_item = grouper["amount"].sum()  # item bazında satış
    trx_item = trx_item.reset_index()  # aylık satış

    # indexlerin oluşturulması. aylık olarak index oluşturulacak.
    date_index = pd.date_range(trx_item.date.min(), trx_item.date.max(), freq="ME")
    dt_i = date_index.to_frame().reset_index(drop=True)
    dt_i.columns = ["date"]
    duos = trx_item[["shop", "item"]].drop_duplicates()
    dt_i["key"] = 0
    duos["key"] = 0
    left = dt_i.merge(duos, on="key", how="outer").drop("key", axis=1)
    left = left.merge(trx_item, on=["date", "shop", "item"], how="outer")

    # yüzde sekseni boş olan dataframelerin drop edilmesi
    drop_eightyp = left.groupby(["shop", "item"])["amount"].count().reset_index()
    drop_eightyp.columns = ["shop", "item", "count"]
    drop_eightyp = drop_eightyp[drop_eightyp["count"] < len(dt_i) * 0.5]
    drop_eightyp.drop(columns=["count"], axis=1, inplace=True)
    drop_eightyp = drop_eightyp.drop_duplicates()

    # drop listte olmayan kayıların alınması.
    merged = left.merge(drop_eightyp, how="outer", on=["shop", "item"],
                        indicator=True)  # iki dataframe mergelenip eşlenmeyenler alınacak.
    unmatched = merged[merged["_merge"] != "both"]
    trx = unmatched.drop(columns=["_merge"], axis=1)

    # son olarak zero olanlara 0.5'i boş olanlar ile concatlıyoruz. bunlara 0 basıcaz.
    zero_preds = pd.concat([zero_preds, drop_eightyp])

    return trx, zero_preds


def model(trx, zero_preds):
    # darts zaman serisinin oluşturulmaı.
    ts = TimeSeries.from_group_dataframe(df=trx,
                                         group_cols=['item', "shop"],
                                         freq="ME",
                                         n_jobs=-1,
                                         verbose=True,
                                         fillna_value=0,
                                         value_cols=["amount"],
                                         time_col="date")  # target column
    # date column not passed index will be used

    # model kurulur
    args = [(index, ts) for index, ts in enumerate(ts)]
    with multiprocessing.Pool(8) as pool:
        results = pool.starmap(utils.seperate_fit_pred, tqdm(args))

    # predictionlar alınır.
    preds = pd.DataFrame(results, columns=["index", "shop", "item", "predicted_amount"])
    preds["predicted_amount"] = round(preds["predicted_amount"], 0)
    preds.loc[preds["predicted_amount"] < 0, "predicted_amount"] = 0 #eğer sıfırdan küçükse 0'a yuvarlanır.
    preds.drop(columns=["index"], axis=1, inplace=True)
    zero_preds.loc[:, "predicted_amount"] = 0

    preds = pd.concat([preds, zero_preds])

    return preds


def pipeline():
    define_paths()
    item_df, cat_df, shop_df, trx_df = get_dataframes()
    trx, zero_preds = data_preprocess(trx_df)
    preds = model(trx, zero_preds)

    return preds
