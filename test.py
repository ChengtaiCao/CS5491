import pickle
import pdb

data_dict = {
    "train_x": 1,
    "train_y": 1,
    "test_x": 1
}

with open("data_dict.pkl", "rb") as f:
    data_dict = pickle.load(f)
    pdb.set_trace()
    print(data_dict)
