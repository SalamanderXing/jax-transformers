from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial


class ReverseDataset(Dataset):
    def __init__(self, num_categories, seq_len, size, np_rng):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size
        self.np_rng = np_rng

        self.data = self.np_rng.integers(
            self.num_categories, size=(self.size, self.seq_len)
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = np.flip(inp_data, axis=0)
        return inp_data, labels


# Combine batch elements (all numpy) by stacking
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_loaders(train_batch_size: int = 128, test_batch_size: int = 128):

    dataset = partial(ReverseDataset, 10, 16)
    rev_train_loader = DataLoader(
        dataset(50000, np_rng=np.random.default_rng(42)),
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=numpy_collate,
    )
    rev_val_loader = DataLoader(
        dataset(1000, np_rng=np.random.default_rng(43)),
        batch_size=test_batch_size,
        collate_fn=numpy_collate,
    )
    rev_test_loader = DataLoader(
        dataset(10000, np_rng=np.random.default_rng(44)),
        batch_size=test_batch_size,
        collate_fn=numpy_collate,
    )
    return rev_train_loader, rev_val_loader, rev_test_loader
