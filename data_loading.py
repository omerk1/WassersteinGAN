import torch
import torchvision
import torchvision.transforms as transforms


def get_mnist_dataset():
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.Scale(32),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    transform = transforms.Compose(
        [
            torchvision.transforms.Scale(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root="data/",
        train=True,
        download=True,
        transform=transform
    )

    return dataset


def filter_mnist_dataset_to_one_label(dataset, label=7):
    idx = dataset.train_labels == label
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    return dataset


def get_mnist_data_loader(batch_size):
    dataset = get_mnist_dataset()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return data_loader