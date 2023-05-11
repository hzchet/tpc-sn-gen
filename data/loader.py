from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from .preprocessing import preprocess_features, read_csv_2d


class MyDataset(Dataset):
    def __init__(self, data, features):
        super().__init__()
        self.data = data
        self.features = features
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.features[index]


def get_loaders(scaler, batch_size=32, data_version=None, pad_range=(40, 50), time_range=(265, 280), strict=False):
    data, raw_features = read_csv_2d(pad_range=pad_range, time_range=time_range, strict=strict)
    data_scaled = scaler.scale(data).astype('float32')
    
    Y_train, Y_test, X_train_raw, X_test_raw = train_test_split(data_scaled, raw_features, test_size=0.25, random_state=42)
    
    train_dataset = MyDataset(Y_train, preprocess_features(X_train_raw.astype('float32')))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, X_test_raw, Y_test