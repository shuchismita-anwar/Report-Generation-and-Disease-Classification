import torch, random
import numpy as np
import torch.utils.data as data

from typing import Dict, Tuple
from .datasets import PhysioNetClincalDataset, ReflacxDataset, collate_fn
from .transforms import get_tensorise_h_flip_transform
from torch.utils.data import DataLoader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader_g(seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

# def get_datasets(
#     dataset_params_dict: Dict,
# ) -> Tuple[ReflacxDataset, ReflacxDataset, ReflacxDataset, ReflacxDataset]:

#     detect_eval_dataset = ReflacxDataset(
#         **{**dataset_params_dict,}, # , "dataset_mode": "unified"
#         transforms=get_tensorise_h_flip_transform(train=False),
#     )

#     train_dataset = ReflacxDataset(
#         **dataset_params_dict, split_str="train", transforms=get_tensorise_h_flip_transform(train=True), 
#     )

#     val_dataset = ReflacxDataset(
#         **dataset_params_dict, split_str="val", transforms=get_tensorise_h_flip_transform(train=False),
#     )

#     test_dataset = ReflacxDataset(
#         **dataset_params_dict, split_str="test", transforms=get_tensorise_h_flip_transform(train=False),
#     )

#     return detect_eval_dataset, train_dataset, val_dataset, test_dataset

def get_datasets(
    dataset_params_dict: Dict,
    using_reflacx = True,
) -> Tuple[data.Dataset, data.Dataset, data.Dataset, data.Dataset]:
    
    if using_reflacx:
        ds = ReflacxDataset
    else:
        ds = PhysioNetClincalDataset

    detect_eval_dataset = ds(
        **{
            **dataset_params_dict,
        },
    )

    train_dataset = ds(
        **dataset_params_dict,
        split_str="train",
        random_flip=True,
    )

    val_dataset = ds(
        **dataset_params_dict,
        split_str="val",
        random_flip=False,
    )

    test_dataset = ds(
        **dataset_params_dict,
        split_str="test",
        random_flip=False,
    )

    return detect_eval_dataset, train_dataset, val_dataset, test_dataset



# def get_datasets(
#     dataset_params_dict: Dict,
# ) -> Tuple[ReflacxDataset, ReflacxDataset, ReflacxDataset, ReflacxDataset]:

#     detect_eval_dataset = ReflacxDataset(
#         **{
#             **dataset_params_dict,
#         },
#     )

#     train_dataset = ReflacxDataset(
#         **dataset_params_dict,
#         split_str="train",
#         random_flip=True,
#     )

#     val_dataset = ReflacxDataset(
#         **dataset_params_dict,
#         split_str="val",
#         random_flip=False,
#     )

#     test_dataset = ReflacxDataset(
#         **dataset_params_dict,
#         split_str="test",
#         random_flip=False,
#     )

#     return detect_eval_dataset, train_dataset, val_dataset, test_dataset


def get_dataloaders(
    train_dataset: data.Dataset,
    val_dataset: data.Dataset,
    test_dataset: data.Dataset,
    batch_size: int = 4,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=get_dataloader_g(seed),
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=get_dataloader_g(seed),
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=get_dataloader_g(seed),
        drop_last=True,  # if we don't make it == True, we may get a batch with only size=1
    )

    return train_dataloader, val_dataloader, test_dataloader
