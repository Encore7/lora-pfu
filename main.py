import yaml

from dataloader import FederatedDataset

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get loaders
data_loader = FederatedDataset(config)
train_loaders, test_loaders = data_loader.get_dataloaders()

# Print shapes of the first batch from train and test loaders
for x, y in train_loaders[0]:
    print("Train batch:", x.shape, y.shape)
    break

for x, y in test_loaders[0]:
    print("Test batch:", x.shape, y.shape)
    break
