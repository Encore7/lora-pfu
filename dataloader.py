import os
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class FederatedDataset:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_name = config["dataset"].lower()
        self.data_dir = config["data_dir"]
        self.num_clients = config["federated"]["num_clients"]
        self.alpha = config["federated"]["iid_dirichlet"]
        self.batch_size = config["federated"]["batch_size"]

        self.train_dataset = None
        self.test_dataset = None
        self.train_partitions = {}
        self.test_partitions = {}

        os.makedirs(self.data_dir, exist_ok=True)

    def _is_dataset_downloaded(self):
        """Check if the specific dataset appears to be downloaded"""
        expected_folders = {
            "mnist": os.path.join(self.data_dir, "MNIST"),
            "fmnist": os.path.join(self.data_dir, "FashionMNIST"),
            "cifar10": os.path.join(self.data_dir, "cifar-10-batches-py"),
        }

        folder = expected_folders.get(self.dataset_name)
        return (
            folder is not None and os.path.isdir(folder) and len(os.listdir(folder)) > 0
        )

    def load_dataset(self):
        """Download and load dataset with proper transforms"""
        if self.dataset_name == "mnist":
            Dataset = datasets.MNIST
            transform = transforms.ToTensor()
        elif self.dataset_name == "fmnist":
            Dataset = datasets.FashionMNIST
            transform = transforms.ToTensor()
        elif self.dataset_name == "cifar10":
            Dataset = datasets.CIFAR10
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        download = not self._is_dataset_downloaded()

        self.train_dataset = Dataset(
            root=self.data_dir, train=True, download=download, transform=transform
        )
        self.test_dataset = Dataset(
            root=self.data_dir, train=False, download=download, transform=transform
        )

    def _dirichlet_partition(self, dataset):
        """Partition dataset labels among clients using Dirichlet distribution"""
        labels = np.array(dataset.targets)
        num_classes = len(np.unique(labels))
        client_indices = defaultdict(list)

        for cls in range(num_classes):
            idx_cls = np.where(labels == cls)[0]
            np.random.shuffle(idx_cls)
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]
            split_indices = np.split(idx_cls, proportions)

            for client_id, idx in enumerate(split_indices):
                client_indices[client_id].extend(idx.tolist())

        return client_indices

    def partition_data(self):
        """Split training and testing data into client-wise subsets"""
        self.train_partitions = self._dirichlet_partition(self.train_dataset)
        self.test_partitions = self._dirichlet_partition(self.test_dataset)

    def _create_client_loaders(self, dataset, partition, shuffle):
        """Create a PyTorch DataLoader for each client"""
        loaders = {}
        for client_id, indices in partition.items():
            subset = Subset(dataset, indices)
            loaders[client_id] = DataLoader(
                subset, batch_size=self.batch_size, shuffle=shuffle
            )
        return loaders

    def get_dataloaders(self):
        """Return train and test DataLoaders per client"""
        self.load_dataset()
        self.partition_data()

        train_loaders = self._create_client_loaders(
            self.train_dataset, self.train_partitions, shuffle=True
        )
        test_loaders = self._create_client_loaders(
            self.test_dataset, self.test_partitions, shuffle=False
        )

        return train_loaders, test_loaders
