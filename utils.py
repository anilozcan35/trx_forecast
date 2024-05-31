import os

from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode


# Outlier sınırlarını belirleyen fonksyion
def outlier_thresholds(dataframe, variable): # OUTLIER HESAPLAMAK İÇİN
    quartile1 = dataframe[variable].quantile(0.25) # EŞİKLER NORMALDE 0.25 ÜZERİNDEN YAPILIRDI NORMALDE PROBLEM BAZINDA UCUNDAN TIRAŞLAMAK İÇİN BÖYLE YAPIYORUZ.
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit # AYKIRI DEĞELERİ BASKILAMAK İÇİN KULLANICAĞIMIZ FONKSYİON

# outlier sınırları ile değerleri değiştiren fonksiyon
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit # AYKIRI DEĞELERİ BASKILAMAK İÇİN KULLANICAĞIMIZ FONKSYİON

def grab_col_names(dataframe, cat_th=10, car_th=20): # SCRİPT BAZINDA DEĞİŞKENLERİ AYIRMAK İÇİN KULLANILAN FONKSYİON
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"] # NUMERİK GÖRÜNEN AMA KATEGORİK FONKSİYONLAR
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"] # ÖLÇÜLEMEZ KATEGORİKLER
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car] # KATEGORİK DEĞİŞKENLERİN SON HALİ

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] # INTEGER VE FLOAT OLANLAR GELECEK
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# KATEGORİK DEĞİŞKENLERİN ÖZETİ
def cat_summary(dataframe, col_name, plot=False): # DEĞİŞKENİN İSMİ VE İLGİLİ DEĞİŞKENİN SINIFLARININ DAĞILIMI
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# NUMERİK DEĞİŞKENLERİN SUMMARYSİ
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    return dataframe[numerical_col].describe(quantiles)

import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
# max_rows almamızı sağlar
pd.set_option('display.max_rows', 5)

def seperate_fit_pred(i, ts):
    serie = ts
    model = ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.ADDITIVE)
    model.fit(serie)
    pred = model.predict(1)
    return (i, ts.static_covariates["shop"].iloc[0].astype("int16"),
            ts.static_covariates["item"].iloc[0].astype("int32"), pred.last_value())

