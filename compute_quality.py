import os
import torch
import numpy as np
import json

from tqdm import tqdm
from datasets import DatasetDict
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error

from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning


filterwarnings(category=ConvergenceWarning, action="ignore")


def main():
    checkpoints = os.listdir("/proj/mechanistic.shadow/mrofin/tinylinguist/models/")

    data = []

    for path in tqdm(checkpoints):
        if path.startswith("checkpoint"):
            for layer in [2, 4, 6, 8]:
                data.append(dict(
                    step=int(path.split("-")[1]),
                    layer=layer,
                    **{
                        k: torch.load(os.path.join("/proj/mechanistic.shadow/mrofin/tinylinguist/models/", path, k + f"_representations_2_layer_{layer}.pt"))
                        for k in ["tc_train", "tc_test", "dl_train", "dl_test"]
                    }
                ))

    labels = DatasetDict.load_from_disk("/proj/mechanistic.shadow/mrofin/tinylinguist/data/val_linguistic_features_2/")

    tc_encoder = LabelEncoder()

    tc_y_train = tc_encoder.fit_transform(labels["tc_train"]["tc_label"])
    tc_y_test = tc_encoder.transform(labels["tc_test"]["tc_label"])

    td_y_train = np.array(labels["dl_train"]["tree_depth"])
    td_y_test = np.array(labels["dl_test"]["tree_depth"])

    sl_y_train = np.array(labels["dl_train"]["length"])
    sl_y_test = np.array(labels["dl_test"]["length"])


    results = []

    for current_preds in tqdm(data):
        tc_X_train = current_preds["tc_train"]
        tc_X_test = current_preds["tc_test"]
        td_sl_X_train = current_preds["dl_train"]
        td_sl_X_test = current_preds["dl_test"]

        tc_model = LogisticRegression(max_iter=1000).fit(tc_X_train, tc_y_train)
        td_model = Ridge().fit(td_sl_X_train, td_y_train)
        sl_model = Ridge().fit(td_sl_X_train, sl_y_train)

        tc_pred = tc_model.predict(tc_X_test)
        td_pred = td_model.predict(td_sl_X_test)
        sl_pred = sl_model.predict(td_sl_X_test)


        results.append(dict(
            step=current_preds["step"],
            layer=current_preds["layer"],
            tc_acc=accuracy_score(tc_y_test, tc_pred),
            td_mse=mean_squared_error(td_y_test, td_pred),
            sl_mse=mean_squared_error(sl_y_test, sl_pred)
        ))
 
    with open("/proj/mechanistic.shadow/mrofin/tinylinguist/data/val_results_2.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
