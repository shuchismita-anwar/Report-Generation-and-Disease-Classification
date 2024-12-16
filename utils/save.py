import os
import numpy as np
from typing import Dict, List, Tuple
import torch
import pickle

from datetime import datetime
from copy import deepcopy
from models.setup import ModelSetup
from utils.detect_utils import MetricLogger
from utils.engine import evaluate
from utils.eval import get_ap_ar, get_performance
import torch.nn as nn

import utils.print as print_f
from models.train import TrainingInfo, get_task_metric_name_from_standard
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset


def get_data_from_metric_logger(loger: MetricLogger) -> Dict[str, float]:
    train_data = {}
    for k in loger.meters.keys():
        train_data[k] = loger.meters[k].avg

    return train_data


###########################################################
# def save_checkpoint(
#     train_info: TrainingInfo,
#     model: nn.Module,
#     val_ar: float,
#     val_ap: float,
#     test_ar: float,
#     test_ap: float,
#     optim: Optimizer = None,
# ) -> TrainingInfo:
#     current_time_string = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

#     model_path = (
#         (
#             f"val_ar_{val_ar:.4f}_ap_{val_ap:.4f}_"
#             + f"test_ar_{test_ar:.4f}_ap_{test_ap:.4f}_"
#             + f"epoch{train_info.epoch}_{train_info.clinical_cond}Clincal_{current_time_string}"
#             + f"_{train_info.model_setup.name}"
#         )
#         .replace(":", "_")
#         .replace(".", "_")
#     )

#     train_info.final_model_path = model_path

#     torch.save(
#         model.state_dict(),
#         os.path.join(os.path.join("trained_models", train_info.final_model_path)),
#     )

#     # Save optimizer if necessary.
#     if optim:
#         torch.save(
#             optim.state_dict(),
#             os.path.join(
#                 os.path.join("trained_models", f"{train_info.final_model_path}_optim")
#             ),
#         )

#     with open(
#         os.path.join("training_records", f"{train_info.final_model_path }.pkl"), "wb",
#     ) as train_info_f:
#         pickle.dump(train_info, train_info_f)

#     return train_info


def save_checkpoint(
    task_str: str,
    metric_str: str,
    train_info: TrainingInfo,
    model: nn.Module,
    val_performance_value: float,
    test_performance_value: float,
    optimizer: Optimizer = None,
    dynamic_weight: nn.Module = None,
) -> TrainingInfo:
    current_time_string = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

    model_path = (
        (
            f"{task_str}_val_{metric_str}_{val_performance_value:.4f}_"
            + f"test_{metric_str}_{test_performance_value:.4f}_"
            + f"epoch{train_info.epoch}_{current_time_string}"
            + f"_{train_info.model_setup.name}"
        )
        .replace(":", "_")
        .replace(".", "_")
    )

    train_info.final_model_path = model_path

    saving_dict = {"model_state_dict": model.state_dict()}
    if optimizer:
        saving_dict["optimizer_state_dict"] = optimizer.state_dict()

    if dynamic_weight:
        saving_dict["dynamic_weight_state_dict"] = dynamic_weight.state_dict()

    os.makedirs("trained_models", exist_ok=True)
    torch.save(
        saving_dict,
        os.path.join(os.path.join("trained_models", train_info.final_model_path)),
    )

    os.makedirs("training_records", exist_ok=True)
    # saving the train_info.
    # remove evaluators to save disk space.
    train_info.remove_evaluators()
    with open(
        os.path.join("training_records", f"{train_info.final_model_path }.pkl"),
        "wb",
    ) as train_info_f:
        pickle.dump(train_info, train_info_f)

    return train_info


def remove_previous_model(previous_model: str):
    if not previous_model is None:
        # delete previous model
        if os.path.exists(os.path.join(os.path.join("trained_models", previous_model))):
            os.remove(os.path.join(os.path.join("trained_models", previous_model)))
        # delete previous training records.
        if os.path.exists(os.path.join("training_records", f"{previous_model}.pkl")):
            os.remove(os.path.join("training_records", f"{previous_model}.pkl"))
        print(f"Previous model: [{previous_model}] has been remove!!")



def check_best(
    setup: ModelSetup,
    train_info: TrainingInfo,
    val_performance_dict,
    eval_params_dict: Dict,
    model: nn.Module,
    optim: Optimizer,
    test_dataloader: DataLoader,
    test_coco: Dataset,
    iou_types: List[str],
    device: str,
    dynamic_weight: nn.Module = None,
    test_performance=None,
) -> Tuple[float, float, TrainingInfo]:

    ## Targeting the model with higher Average Recall and Average Precision.
    is_better = []
    for s in setup.performance_standards:
        name = get_task_metric_name_from_standard(s)
        val_performance_value = val_performance_dict[name]
        if not name in train_info.best_val_performance:
            train_info.best_val_performance[name] = -np.inf

        if not name in train_info.best_performance_model_path:
            train_info.best_performance_model_path[name] = None

        is_better.append(val_performance_value> train_info.best_val_performance[name])

    ### check if any is better?
    if any(is_better):
        if test_performance is None:
            train_info.test_evaluator, _ = evaluate(
                setup=setup,
                model=model,
                data_loader=test_dataloader,
                device=device,
                params_dict=eval_params_dict,
                coco=test_coco,
                iou_types=iou_types,
                return_dt_gt=True,
            )

            test_performance = get_performance(
                dataset=test_dataloader.dataset,
                all_tasks=list(model.task_performers.keys()),
                evaluator=train_info.test_evaluator,
            )

        test_performance_dict = {
                get_task_metric_name_from_standard(s): test_performance[s["task"]][s["metric"]]
                for s in setup.performance_standards
        }

        for s in setup.performance_standards:
            name = get_task_metric_name_from_standard(s)

            if val_performance_dict[name] >  train_info.best_val_performance[name]:
                test_performance_value = test_performance_dict[name]
                val_performance_value = val_performance_dict[name]
                previous_best_model = deepcopy(train_info.best_performance_model_path[name])
                train_info = save_checkpoint(
                    task_str=s['task'],
                    metric_str=s['metric'],
                    train_info=train_info,
                    model=model,
                    val_performance_value=val_performance_value,
                    test_performance_value=test_performance_value,
                    optimizer=optim,
                    dynamic_weight=dynamic_weight,
                )
                train_info.best_performance_model_path[name] = train_info.final_model_path
                train_info.best_val_performance[name] = val_performance_dict[name]
                remove_previous_model(previous_best_model)

    return val_performance_dict, train_info



# def check_best(
#     setup: ModelSetup,
#     train_info: TrainingInfo,
#     val_performance_value,
#     eval_params_dict: Dict,
#     model: nn.Module,
#     optim: Optimizer,
#     test_dataloader: DataLoader,
#     test_coco: Dataset,
#     iou_types: List[str],
#     device: str,
#     dynamic_weight: nn.Module = None,
#     test_performance=None,
# ) -> Tuple[float, float, TrainingInfo]:

#     ## Targeting the model with higher Average Recall and Average Precision.
    
#     if val_performance_value > train_info.best_val_performance:
#         if test_performance is None:
#             train_info.test_evaluator, _ = evaluate(
#                 setup=setup,
#                 model=model,
#                 data_loader=test_dataloader,
#                 device=device,
#                 params_dict=eval_params_dict,
#                 coco=test_coco,
#                 iou_types=iou_types,
#                 return_dt_gt=True,
#             )

#             test_performance = get_performance(
#                 dataset=test_dataloader.dataset,
#                 all_tasks=list(model.task_performers.keys()),
#                 evaluator=train_info.test_evaluator,
#             )

#         test_performance_value = test_performance[setup.performance_standard_task][
#                 setup.performance_standard_metric
#             ]  # get_ap_ar(train_info.test_evaluator['lesion-detection'])

#         if val_performance_value > train_info.best_val_performance:
#             ## Save best validation model
#             previous_ar_model = deepcopy(train_info.best_performance_model_path)
#             train_info = save_checkpoint(
#                 setup=setup,
#                 train_info=train_info,
#                 model=model,
#                 val_performance_value=val_performance_value,
#                 test_performance_value=test_performance_value,
#                 optimizer=optim,
#                 dynamic_weight=dynamic_weight,
#             )
#             train_info.best_performance_model_path = train_info.final_model_path
#             train_info.best_val_performance = val_performance_value
#             remove_previous_model(previous_ar_model)

#     return val_performance_value, train_info


def end_train(
    setup: ModelSetup,
    train_info: TrainingInfo,
    model: nn.Module,
    optim: Optimizer,
    eval_params_dict: Dict,
    last_val_performance: float,
    test_dataloader: DataLoader,
    device: str,
    test_coco: Dataset,
    iou_types: List[str],
    dynamic_weight: nn.Module = None,
    test_performance=None,
) -> TrainingInfo:

    train_info.timer.end_training()
    sec_took = train_info.timer.has_took_sec()

    print_f.print_title(
        f"| Training Done, start testing! | [{train_info.epoch}] Epochs Training time: [{sec_took}] seconds, Avg time / Epoch: [{sec_took/train_info.epoch}] seconds"
    )

    # print model
    if train_info.model_setup.save_early_stop_model:
        print_f.print_title(
            f"Best Performance model has been saved to: [{train_info.best_performance_model_path}]"
        )

    if test_performance is None:
        train_info.test_evaluator, test_logger = evaluate(
            setup=setup,
            model=model,
            data_loader=test_dataloader,
            device=device,
            params_dict=eval_params_dict,
            coco=test_coco,
            iou_types=iou_types,
            return_dt_gt=True,
        )

        test_performance = get_performance(
            dataset=test_dataloader.dataset,
            all_tasks=list(model.task_performers.keys()),
            evaluator=train_info.test_evaluator,
        )

    s = setup.performance_standards[0]
    test_performance_value = test_performance[s['task']][
        s['metric']
    ]
    
    train_info = save_checkpoint(
                task_str=s['task'],
                metric_str=s['metric'],
                train_info=train_info,
                model=model,
                val_performance_value=last_val_performance,
                test_performance_value=test_performance_value,
                optimizer=optim,
                dynamic_weight=dynamic_weight,
    )

    print_f.print_title(
        f"The final model has been saved to: [{train_info.final_model_path}]"
    )
    return train_info



# def end_train(
#     setup: ModelSetup,
#     train_info: TrainingInfo,
#     model: nn.Module,
#     optim: Optimizer,
#     eval_params_dict: Dict,
#     last_val_performance: float,
#     test_dataloader: DataLoader,
#     device: str,
#     test_coco: Dataset,
#     iou_types: List[str],
#     dynamic_weight: nn.Module = None,
#     test_performance=None,
# ) -> TrainingInfo:

#     train_info.timer.end_training()
#     sec_took = train_info.timer.has_took_sec()

#     print_f.print_title(
#         f"| Training Done, start testing! | [{train_info.epoch}] Epochs Training time: [{sec_took}] seconds, Avg time / Epoch: [{sec_took/train_info.epoch}] seconds"
#     )

#     # print model
#     if train_info.model_setup.save_early_stop_model:
#         print_f.print_title(
#             f"Best Performance model has been saved to: [{train_info.best_performance_model_path}]"
#         )

#     if test_performance is None:
#         train_info.test_evaluator, test_logger = evaluate(
#             setup=setup,
#             model=model,
#             data_loader=test_dataloader,
#             device=device,
#             params_dict=eval_params_dict,
#             coco=test_coco,
#             iou_types=iou_types,
#             return_dt_gt=True,
#         )

#         test_performance = get_performance(
#             dataset=test_dataloader.dataset,
#             all_tasks=list(model.task_performers.keys()),
#             evaluator=train_info.test_evaluator,
#         )

#     test_performance_value = test_performance[setup.performance_standard_task][
#         setup.performance_standard_metric
#     ]

#     train_info = save_checkpoint(
#         setup=setup,
#         train_info=train_info,
#         model=model,
#         val_performance_value=last_val_performance,
#         test_performance_value=test_performance_value,
#         optimizer=optim,
#         dynamic_weight=dynamic_weight,
#     )

#     print_f.print_title(
#         f"The final model has been saved to: [{train_info.final_model_path}]"
#     )
#     return train_info
