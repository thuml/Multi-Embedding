import os.path as osp
import numpy as np
import pandas as pd
import torch.utils.data
import random
import math


class Amazon(torch.utils.data.Dataset):
    """
    """

    def __init__(self, data, split, field_dims):
        data.sort_values('timestamp', inplace=True)

        timestamps = data.timestamp.to_numpy()
        if split == 'train':
            start_timestamp = 0
            end_timestamp = timestamps[int(0.8 * len(data))]
        elif split == 'val':
            start_timestamp = timestamps[int(0.8 * len(data))]
            end_timestamp = timestamps[int(0.9 * len(data))]
        else:
            start_timestamp = timestamps[int(0.9 * len(data))]
            end_timestamp = math.inf
        # print(split, start_timestamp, end_timestamp)

        def gen_neg(pos_list):
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, field_dims[1] - 1)
            return neg

        self.items = []
        self.targets = []
        for user_id, data in data.groupby("user_id"):
            if len(data) < 5:
                continue
            item_ids = data.to_numpy()[:, 1]
            timestamps = data.to_numpy()[:, 3]
            for item_id, timestamp in zip(item_ids, timestamps):
                if start_timestamp <= timestamp < end_timestamp:
                    self.items.append([user_id, item_id])
                    self.targets.append(1)
                    self.items.append([user_id, gen_neg(item_ids)])
                    self.targets.append(0)

        self.items = np.array(self.items).astype(np.int)
        self.targets = np.array(self.targets).astype(np.float32)
        self.field_dims = field_dims
        self.user_field_idx = np.array((0,), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]


def get_amazon_datasets(root, domains):
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {
        "user_id": LabelEncoder(),
        "item_id": LabelEncoder()
    }
    data = {}
    for domain in domains:
        # TODO add sorting preprocessing
        data[domain] = pd.read_csv(osp.join(root, "ratings_{}.csv".format(domain)),
                           names=["user_id", "item_id", "label", "timestamp"])
    all_data = pd.concat(data.values())
    field_dims = []
    for feat in label_encoders:
        label_encoders[feat].fit(all_data[feat])
        field_dims.append(len(label_encoders[feat].classes_))

    for domain in domains:
        for feat in label_encoders:
            data[domain][feat] = label_encoders[feat].transform(data[domain][feat])

    return {domain: Amazon(data[domain], 'train', field_dims) for domain in domains},\
           {domain: Amazon(data[domain], 'val', field_dims) for domain in domains}, \
            {domain: Amazon(data[domain], 'test', field_dims) for domain in domains}
