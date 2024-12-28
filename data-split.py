import os
import pandas as pd

data_path = " write your csv file path"
data = pd.read_csv("./data/btcusd_1-min_data.csv")
data.head()


sv_fd = "./data/"


train_data = data.sample(frac=0.8, random_state=100).reset_index(drop=True)
test_data = data.drop(train_data.index).reset_index(drop=True)


print("train_data for Modeling: " + str(train_data.shape))
print("test_data for Predictions: " + str(test_data.shape))


train_data.to_csv(os.path.join(sv_fd, "train.csv"))
test_data.to_csv(os.path.join(sv_fd, "test.csv"))
