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
import src.lstm_dataprep as lstm_dataprep

importlib.reload(utils)


#Hata metriği olarak custom bir metrik kullanmak çok daha doğru olacaktır.
#Bu metrik root(abs(y_real - ynaive)* price) / root(abs(y_real - y_lstm_pred)*price) olabilir.
#Eğer Hata 1'in üzerinde gelirse. Aslında naive forecastten ileri gidememiş oluruz.
#Eğer Hata 1'in altında gelirse naive forecastten daha iyi bir forecast yapmışızdır.
#Bu forecast hatası içerisinde priceın da etkisi saklıdır.

def model(ts):
    # timeseries train - validation split
    #ts_train = [serie[:-1] for serie in ts]
    #ts_val = [serie[-1] for serie in ts]

    # scaling
    transformer = Scaler()
    train_transformed = transformer.fit_transform(ts)
    #val_transformed = transformer.transform(ts_val)

    # Model
    my_model = RNNModel(
        model="LSTM",
        hidden_dim=5,
        dropout=0,
        batch_size=16,
        n_epochs=2,
        optimizer_kwargs={"lr": 1e-3},
        model_name="LSTM_TRX",
        log_tensorboard=True,
        random_state=42,
        training_length=30,
        input_chunk_length=1,
        force_reset=True,
        save_checkpoints=True,
    )

    # prediction alınması ve inverse transform
    my_model.fit(train_transformed, verbose=True)

    CURRENT_DIR = os.getcwd()
    PARRENT_DIR = os.path.dirname(CURRENT_DIR)

    my_model.save(os.path.join(PARRENT_DIR, "model\\lstm.pt"))
    # with open(os.path.join(PARRENT_DIR, "model/ltsm.pkl"), 'wb') as f:
    #     pickle.dump(my_model, f)

    return my_model, transformer, train_transformed


def get_cat_preds(ts):
    # modelin olup olmadığına göre prediction yapılacak veya model eğitilip prediction yapılacak.
    CURRENT_DIR = os.getcwd()
    PARRENT_DIR = os.path.dirname(CURRENT_DIR)
    print(PARRENT_DIR)
    LSTM_PATH = os.path.join(PARRENT_DIR, "model\\lstm.pt")
    print(LSTM_PATH)

    # model var mı kontrol et yoksa eğit ve döndür.
    if os.path.isfile(LSTM_PATH):
        print("if")
        lstm = RNNModel.load(LSTM_PATH)
        print("lstm bitis")
        # with open(LSTM_PATH, 'rb') as f:
        #     my_model = pickle.load(f)
        transformer = Scaler()
        train_transformed = transformer.fit_transform(ts)
    else:
        print("else")
        lstm, transformer, train_transformed = model(ts)

    # predictionları al.
    print("pred baslangıc")
    preds = lstm.predict(1, series=train_transformed)
    # inverse scaling
    preds = transformer.inverse_transform(preds)
    # predictionları pandasa çeviriyoruz
    preds_pd = [pred.pd_dataframe() for pred in preds]
    # static covariatesler pandasa çevirliyor
    statics = [pred.static_covariates_values()[0] for pred in preds]
    statics_covariates = pd.DataFrame(statics, columns=["shop", "item_category_id", "cluster"])
    preds = pd.concat(preds_pd)

    # kategori bazında sonuçların olduğu nihai df
    pred_df = pd.merge(statics_covariates, preds.reset_index(drop=True), left_index=True, right_index=True)
    pred_df["shop"] = pred_df["shop"].astype("int16")
    pred_df["item_category_id"] = pred_df["item_category_id"].astype("int16")

    return pred_df


def prediction_item_basis(pred_df, ratios):
    # predictionlar ve ratioların oldugu df
    ratios_preds = ratios.merge(pred_df, left_on="item_category_id", right_on="item_category_id", how="inner")
    # aslında burada bir shopta o item satılmıyorsa da ratiodan ötürü geliyor gibi bi durum var. İleride 0 basılacak olduğu için bir kaçak olamyacaktır.
    # ya da daha isabetli bir tahmin için store-item ikilelerinin kategorilerdeki ağırlıkları hesaplanarak devam edilebilir.
    # bu ağırlıklar ile kategori bazındaki tahminler çarpılarak store-item bazına inilebilir.

    # item bazında predictionlar
    ratios_preds["predicted_amount"] = ratios_preds["ratio"] * ratios_preds["amount"]

    # item bazında predictionlar alınıyor.
    item_basis_preds = ratios_preds[["item", "shop", "predicted_amount"]]
    item_basis_preds.loc[:, "predicted_amount"] = round(item_basis_preds["predicted_amount"], 0)

    return item_basis_preds


def test_predictions(item_basis_preds):
    CURRENT_DIR = os.getcwd()
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    TEST_PATH = os.path.join(ROOT_DIR, "data\\test.csv")

    # test_df'in içeri alınması ve predictionlar ile joinlenmesi
    test_df = pd.read_csv(TEST_PATH, index_col=0)
    test_preds = test_df.merge(item_basis_preds, on=["item", "shop"], how="inner")
    test_preds["predicted_amount"] = round(test_preds["predicted_amount"], 0)

    return test_preds

def pipeline():
    ts, ratios = lstm_dataprep.pipeline()
    print("1")
    pred_df = get_cat_preds(ts)
    print("2")
    item_basis_preds = prediction_item_basis(pred_df, ratios)
    print("3")
    test_preds = test_predictions(item_basis_preds)
    print("4")
    return test_preds


if __name__ == "__main__":
    ts, ratios = lstm_dataprep.pipeline()
    print("1")
    pred_df = get_cat_preds(ts)
    print("2")
    item_basis_preds = prediction_item_basis(pred_df, ratios)
    print("3")
    test_preds = test_predictions(item_basis_preds)
    print("4")
    print(test_preds.head())

# # timeseries train - validation split
# ts_train = [serie[:-1] for serie in ts]
# ts_val = [serie[-1] for serie in ts]
#
# # scaling
# transformer = Scaler()
# train_transformed = transformer.fit_transform(ts_train)
# val_transformed = transformer.transform(ts_val)
#
# # Model
# my_model = RNNModel(
#     model="LSTM",
#     hidden_dim=5,
#     dropout=0,
#     batch_size=16,
#     n_epochs=5,
#     optimizer_kwargs={"lr": 1e-3},
#     model_name="LSTM_TRX",
#     log_tensorboard=True,
#     random_state=42,
#     training_length=30,
#     input_chunk_length=1,
#     force_reset=True,
#     save_checkpoints=True,
# )
#
# # prediction alınması ve inverse transform
# my_model.fit(train_transformed, verbose=True)
# preds = my_model.predict(1, series=train_transformed)
# # preds = transformer.inverse_transform(preds)
