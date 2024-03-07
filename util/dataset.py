import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.low_text_feature = torch.from_numpy(np.array(dataset['input_ids']))
        self.middle_text_feature = torch.from_numpy(np.array(dataset['attention_mask']))
        self.high_text_feature = torch.from_numpy(np.array(dataset['token_type_ids']))
        self.text_simi_image_feature = torch.from_numpy(np.array(dataset['text_simi_image_feature']))
        self.similarity = torch.from_numpy(np.array(dataset['similarity']))
        self.image = list(dataset['image'])
        self.label = torch.from_numpy(np.array(dataset['label']))
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return (self.low_text_feature[idx], self.middle_text_feature[idx], self.high_text_feature[idx],
                self.text_simi_image_feature[idx], self.similarity[idx], self.image[idx]), self.label[idx]
