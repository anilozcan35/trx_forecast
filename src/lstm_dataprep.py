import os
import importlib
import pickle
import pandas as pd
import utils
import matplotlib.pyplot as plt
from darts import TimeSeries
import darts
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel

importlib.reload(utils)

def define_paths():
    CURRENT_DIR = os.getcwd()
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    DATA_PATH = os.path.join(BASE_DIR, 'data') # pathler kontrol edilmeli.
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


def preprocess_data(item_df, cat_df, shop_df, trx_df):
    # trx_cat ile item_df mergelenir
    trx_cat = trx_df.merge(item_df, left_on="item", right_on="item_id", how="inner")
    trx_cat.drop(columns=["item_id", "item_name", "price", "item_name", "item"], axis=1, inplace=True)

    # trx_cat date kolonu üzerinde düzenlemeler yapılır.
    trx_cat["date"] = pd.to_datetime(trx_cat["date"], format="%d.%m.%Y")
    trx_cat = trx_cat.set_index("date")

    # shop ve item_cat üzerinde gruplama yapılır.
    grouper = trx_cat.groupby([pd.Grouper(freq='ME'), 'shop', "item_category_id"])
    trx_cat_sum = grouper["amount"].sum()
    trx_cat_sum = trx_cat_sum.reset_index()  # model datası

    # son 1 yılda satış yapılan miktarlara göre veriyi clusterlar haline bölücez
    cluster_df = trx_cat_sum[trx_cat_sum["date"] > "2014-09-30"].groupby("item_category_id")[
        "amount"].sum()  # tarih dinamik olmalı.
    cluster_df = cluster_df.reset_index()

    # cluster4'ün oluşturulması
    low_limit, upper_limit = utils.outlier_thresholds(cluster_df, "amount")
    # cluster_df.plot(kind="box")
    # plt.show()
    cluster5 = cluster_df[cluster_df["amount"] > upper_limit]

    # cluster 1-2-3-4 oluşturulması.
    cluster_df.subtract(cluster5)

    # clustera atanmayanların alınması.
    i1 = pd.MultiIndex.from_frame(cluster_df)
    i2 = pd.MultiIndex.from_frame(cluster5)
    unclustered_df = cluster_df[~i1.isin(i2)]

    unclustered_df["cluster"] = pd.qcut(unclustered_df["amount"], 4, labels=[1, 2, 3, 4])
    cluster5.loc[:, "cluster"] = 5

    # cluster dataframelerini bir listeye alıp.
    clusters = [unclustered_df, cluster5]

    # concat ediyoruz.
    clusters = pd.concat(clusters)
    # clusters.drop(columns=["amount"], inplace=True)

    # itemların kategorilerde oluşturdukları satış ağırlıkları(miktar)
    # model kategori bazında kurulacağı için bu ağırlıklara göre tahminler, item kırılımına yansıtılacak.
    trx_item_temp = trx_df.drop(columns=["price"], axis=1)
    trx_item_temp["date"] = pd.to_datetime(trx_item_temp["date"], format="%d.%m.%Y")  # date convert
    trx_item_temp = trx_item_temp.set_index("date")  # set index date
    grouper = trx_item_temp.groupby([pd.Grouper(freq='ME'), 'shop', "item"])  # item bazında grupla
    trx_item = grouper["amount"].sum()  # item bazında satış
    trx_item = trx_item.reset_index()  # aylık satış

    # item satışlar son 1 yıl
    item_total_sale = trx_item[trx_item["date"] > "2014-09-30"].groupby("item")["amount"].sum()  # tarih burada da dinamik olmalı.
    item_total_sale = item_total_sale.reset_index()

    # kategori satışlar son 1 yıl
    trx_cat = trx_cat.reset_index()
    cat_sales_temp = trx_cat[trx_cat["date"] > "2014-09-30"].groupby("item_category_id")["amount"].sum()
    cat_sales = cat_sales_temp.reset_index()

    # itemların kategoriler üzerinde ne ağırlıkla satış yaptıkları hesaplanır.
    # item_total_sale ve cat_salesler mergelenip bir oran alınacak.
    ratios_temp = item_total_sale.merge(item_df, left_on="item", right_on="item_id", how="inner").drop(
        columns=["item_id", "item_name"], axis=1)
    ratios = ratios_temp.merge(cat_sales, on="item_category_id", how="inner")
    ratios.columns = ["item", "item_amount", "item_category_id", "cat_amount"]
    ratios["ratio"] = ratios["item_amount"] / ratios["cat_amount"]
    # checker = ratios.groupby("item_category_id")["ratio"].sum() == 1 olmalı

    # TRX datasının oluşturulması
    clusters.drop(columns=["amount"], axis=1, inplace=True)
    model_data = trx_cat_sum.merge(clusters, left_on="item_category_id", right_on="item_category_id", how="inner")

    return model_data, ratios


def create_ts_object(model_data):
    # tarih dataframei oluşturuyor
    date_index = pd.date_range(model_data.date.min(), model_data.date.max(), freq="ME")
    dt_i = date_index.to_frame().reset_index(drop=True)
    dt_i.columns = ["date"]
    duos = model_data[["shop", "item_category_id", "cluster"]]
    duos.drop_duplicates(inplace=True)
    dt_i["key"] = 0
    duos["key"] = 0
    left = dt_i.merge(duos, on="key", how="outer").drop("key", axis=1)
    model_data = left.merge(model_data, on=["date", "shop", "item_category_id", "cluster"], how="outer")

    # Darts Data Yapısının Hazırlanması
    ts = TimeSeries.from_group_dataframe(df=model_data,
                                         group_cols=['shop', "item_category_id"],
                                         static_cols=["cluster"],
                                         freq="ME",
                                         n_jobs=-1,
                                         verbose=True,
                                         fillna_value=0,
                                         value_cols=["amount"], # target_column
                                         time_col="date") # date column not passed index will be used

    return ts



def pipeline():
    item_df, cat_df, shop_df, trx_df = get_dataframes()
    model_data, ratios = preprocess_data(item_df, cat_df, shop_df, trx_df)
    ts = create_ts_object(model_data)
    print("dataprep finished")
    return ts, ratios


if __name__ == '__main__':
    ROOT_PATH = os.getcwd()
    print(ROOT_PATH)
    pipeline()


