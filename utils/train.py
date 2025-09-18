import torch
import os
import pickle

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, MultiStepLR

from models.setup import ModelSetup
from models.dynamic_loss import DynamicWeightedLoss

from .coco_eval import get_eval_params_dict
from .coco_utils import get_cocos

from datetime import datetime


def get_optimiser(params, setup: ModelSetup) -> Optimizer:
    if setup.optimiser == "adamw":
        print(f"Using AdamW as optimizer with lr={setup.lr}")
        optimiser = torch.optim.AdamW(
            params,
            lr=setup.lr,
            betas=(0.9, 0.999),
            weight_decay=setup.weight_decay,
        )

    elif setup.optimiser == "sgd":
        print(f"Using SGD as optimizer with lr={setup.lr}")
        optimiser = torch.optim.SGD(
            params,
            lr=setup.lr,
            momentum=setup.sgb_momentum,
            weight_decay=setup.weight_decay,
        )
    else:
        raise Exception(f"Unsupported optimiser {setup.optimiser}")

    return optimiser


def get_lr_scheduler(optimizer: Optimizer, setup: ModelSetup) -> _LRScheduler:

    if setup.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=setup.reduceLROnPlateau_factor,
            patience=setup.reduceLROnPlateau_patience,
            min_lr=1e-10,
        )
    elif setup.lr_scheduler == "MultiStepLR":
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones=setup.multiStepLR_milestones,
            gamma=setup.multiStepLR_gamma,
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def num_params(model):
    return sum([param.nelement() for param in model.parameters()])


def print_params_setup(model):
    print(f"[model]: {num_params(model):,}")
    # print(f"[model.backbone]: {num_params(model.backbone):,}")
    # print(f"[model.rpn]: {num_params(model.rpn):,}")
    # print(f"[model.roi_heads]: {num_params(model.roi_heads):,}")
    # print(f"[model.roi_heads.box_head]: {num_params(model.roi_heads.box_head):,}")
    # print(
    #     f"[model.roi_heads.box_head.fc6]: {num_params(model.roi_heads.box_head.fc6):,}"
    # )
    # print(
    #     f"[model.roi_heads.box_head.fc7]: {num_params(model.roi_heads.box_head.fc7):,}"
    # )
    # print(
    #     f"[model.roi_heads.box_predictor]: {num_params(model.roi_heads.box_predictor):,}"
    # )

    # if hasattr(model.roi_heads, "mask_head") and not model.roi_heads.mask_head is None:
    #     print(f"[model.roi_heads.mask_head]: {num_params(model.roi_heads.mask_head):,}")

    # if hasattr(model, "clinical_convs") and not model.clinical_convs is None:
    #     print(f"[model.clinical_convs]: {num_params(model.clinical_convs):,}")

    # if hasattr(model, "fuse_convs") and not model.fuse_convs is None:
    #     print(f"[model.fuse_convs]: {num_params(model.fuse_convs):,}")


def get_coco_eval_params(
    source_name,
    task_name,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    detect_eval_dataset,
    iou_thrs,
    maxDets,
    use_iobb,
):

    df_path = train_dataloader.dataset.df_path
    file_name = df_path.split(".")[0]
    file_name = f"{file_name}_source_{source_name}_task_{task_name}"
    os.makedirs("./coco_eval_params", exist_ok=True)
    store_path = os.path.join("./coco_eval_params", file_name)

    if os.path.isfile(store_path):
        with open(store_path, "rb") as handle:
            save_dict = pickle.load(handle)

        train_coco = save_dict["train_coco"]
        val_coco = save_dict["val_coco"]
        test_coco = save_dict["test_coco"]
        eval_params_dict = save_dict["eval_params_dict"]

    else:
        train_coco, val_coco, test_coco = get_cocos(
            source_name=source_name,
            task_name=task_name,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
        )

        eval_params_dict = get_eval_params_dict(
            detect_eval_dataset,
            iou_thrs=iou_thrs,
            maxDets=maxDets,
            use_iobb=use_iobb,
        )

        save_dict = {
            "train_coco": train_coco,
            "val_coco": val_coco,
            "test_coco": test_coco,
            "eval_params_dict": eval_params_dict,
        }

        with open(store_path, "wb") as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_coco, val_coco, test_coco, eval_params_dict


def get_dynamic_loss(loss_keys, device):
    dynamic_loss_weight = DynamicWeightedLoss(keys=loss_keys)
    dynamic_loss_weight.to(device)

    return dynamic_loss_weight


def get_params(model, dynamic_loss_weight):
    params = [p for p in model.parameters() if p.requires_grad]
    if dynamic_loss_weight:
        params += [p for p in dynamic_loss_weight.parameters() if p.requires_grad]
    return params
