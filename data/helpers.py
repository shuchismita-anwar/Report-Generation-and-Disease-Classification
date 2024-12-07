from typing import Any, Dict
import torch
from collections import ChainMap


def map_target_to_device(target: Dict, device: str) -> Dict:
    key_value_pairs = []

    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            key_value_pairs.append({k: v.to(device)})
        else:
            key_value_pairs.append({k: v})

    return dict(ChainMap(*key_value_pairs))


def map_dict_to_device(target: Dict, device: str) -> Dict:
    key_value_pairs = []

    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            key_value_pairs.append({k: v.to(device)})
        else:
            key_value_pairs.append({k: v})

    return dict(ChainMap(*key_value_pairs))


def target_processing(target: Dict) -> Dict:
    key_value_pairs = []

    for k, v in target.items():
        key_value_pairs.append({k: v})

    return dict(ChainMap(*key_value_pairs))


def map_dict_elements_to_device(x, device):
    for k in x.keys():
        try:
            x[k] = x[k].to(device)
        except:
            pass

    return x


def map_to_device_for(k_1, k_2, x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        out_dict = {}
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                out_dict.update({k: v.to(device)})
            else:
                out_dict.update({k: v})
        return out_dict
    else:
        return x


def map_2l_nest_dict_to_device(x, device):
    for k_1 in x.keys():
        for k_2 in x[k_1].keys():

            if isinstance(x[k_1][k_2], torch.Tensor):
                x[k_1][k_2] = x[k_1][k_2].to(device)

            elif isinstance(x[k_1][k_2], list):

                try:
                    # list of tensor
                    x[k_1][k_2] = torch.stack(x[k_1][k_2]).to(device)
                except:
                    # list of tensor with different size
                    x[k_1][k_2] = [
                        map_to_device_for(k_1, k_2, e, device) for e in x[k_1][k_2]
                    ]

            elif isinstance(x[k_1][k_2], dict):
                out_dict = {}
                for k, v in x[k_1][k_2].items():
                    if isinstance(v, torch.Tensor):
                        out_dict.update({k: v.to(device)})
                    else:
                        out_dict.update({k: v})
                x[k_1][k_2] = out_dict

            else:
                print(f"{k_1}_{k_2} Not mapped")

    return x


def map_every_thing_to_device(x: Any, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [map_every_thing_to_device(x_i, device) for x_i in x]
    elif isinstance(x, dict):
        return {k: map_every_thing_to_device(v, device) for k, v in x.items()}
    else:
        return x

