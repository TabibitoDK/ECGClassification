import torch
from torch.utils.data import Dataset
from collections import Counter

if __package__:
    from .preprocess import SCGGenerator, Digitizer
else:  # allow running as standalone script if needed
    from preprocess import SCGGenerator, Digitizer


class ECGDataset(Dataset):
    def __init__(self, df_type, transform = None):
        self.transform = transform
        self.scg_generator = SCGGenerator(df_type)
        self.label_list = [self.scg_generator.get_y(idx) for idx in range(len(self))]
    
    def __len__(self):
        return len(self.scg_generator)

    def __getitem__(self, idx):
        X, _ = self.scg_generator.get_X(idx)
        y = self.scg_generator.get_y(idx)
        return torch.tensor(X, dtype = torch.float32).unsqueeze(0), torch.tensor(y, dtype = torch.long)
    
    def get_weights(self):
        weights = []
        counts = Counter(self.label_list)
        class_counts = torch.tensor([count for (_, count) in sorted(counts.items())], dtype = torch.float)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.max()
        for idx in range(len(self)):
            weights.append(1.0 / counts.get(self.label_list[idx], 0))
        return torch.DoubleTensor(weights), class_weights


class GANDataset(Dataset):
    def __init__(self, df_type, transform = None):
        self.transform = transform
        self.digitizer = Digitizer(df_type)
        self.label_list = [self.digitizer.get_y(idx) for idx in range(len(self))]

    def __len__(self):
        return len(self.digitizer)

    def __getitem__(self, idx):
        X = self.digitizer.get_X(idx)
        y = self.digitizer.get_y(idx)
        return torch.tensor(X, dtype = torch.float32).unsqueeze(-1), torch.tensor(y, dtype = torch.long)
