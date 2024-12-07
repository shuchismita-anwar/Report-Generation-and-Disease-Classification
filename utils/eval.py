import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from typing import List
from collections import OrderedDict

from sklearn.metrics import roc_auc_score

from data.strs import TaskStrs
import utils.print as print_f
from models.load import get_model_name, get_trained_model
from data.constants import DEFAULT_REFLACX_LABEL_COLS

# from utils.plot import plot_losses, plot_train_val_evaluators
from utils.train import num_params
from coco_froc_analysis.froc.froc_curve import get_froc_curve, get_interpolate_froc


def find_match_stat(stats, ap, iouThr, areaRng, maxDets):
    if iouThr is None:
        return next(
            (
                item["mean_s"]
                for item in stats
                if (item["ap"] == ap)
                and (item["iouThr"] is None)
                and (item["areaRng"] == areaRng)
                and (item["maxDets"] == maxDets)
            ),
            -1,
        )
    else:
        return next(
            (
                item["mean_s"]
                for item in stats
                if (item["ap"] == ap)
                and (item["iouThr"] == iouThr)
                and (item["areaRng"] == areaRng)
                and (item["maxDets"] == maxDets)
            ),
            -1,
        )


def get_ap_ar(
    evaluator,
    iouThr=None,
    areaRng="all",
    maxDets=30,
):
    # what if we just use

    ap = find_match_stat(
        stats=evaluator.stats,
        ap=1,
        iouThr=iouThr,
        areaRng=areaRng,
        maxDets=maxDets,
    )

    ar = find_match_stat(
        stats=evaluator.stats,
        ap=0,
        iouThr=iouThr,
        areaRng=areaRng,
        maxDets=maxDets,
    )

    # ap = external_summarize(
    #     evaluator.coco_eval["bbox"],
    #     ap=1,
    #     iouThr=iouThr,
    #     areaRng=areaRng,
    #     maxDets=maxDets,
    #     print_result=False,
    # )

    # ar = external_summarize(
    #     evaluator.coco_eval["bbox"],
    #     ap=0,
    #     iouThr=iouThr,
    #     areaRng=areaRng,
    #     maxDets=maxDets,
    #     print_result=False,
    # )

    return {"ap": ap, "ar": ar}


# def save_iou_results(evaluator: CocoEvaluator, suffix: str, model_path: str):
#     os.makedirs("./eval_results", exist_ok=True)

#     ap_ar_dict = OrderedDict(
#         {thrs: [] for thrs in evaluator.coco_eval["bbox"].params.iouThrs}
#     )

#     for thrs in evaluator.coco_eval["bbox"].params.iouThrs:
#         test_ap_ar = get_ap_ar(
#             evaluator,
#             areaRng="all",
#             maxDets=10,
#             iouThr=thrs,
#         )

#         ap_ar_dict[thrs].append(test_ap_ar)

#         print(
#             f"IoBB [{thrs:.4f}] | AR [{test_ap_ar['ar']:.4f}] | AP [{test_ap_ar['ap']:.4f}]"
#         )

#     with open(
#         os.path.join("eval_results", f"{model_path}_{suffix}.pkl"),
#         "wb",
#     ) as training_record_f:
#         pickle.dump(ap_ar_dict, training_record_f)


def get_thrs_evaluation_df(
    models, dataset, disease="all", iobb_thrs=0.5, score_thrs=0.05
):
    all_models_eval_data = {}
    for select_model in models:
        with open(
            os.path.join(
                "eval_results",
                f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
            ),
            "rb",
        ) as f:
            eval_data = pickle.load(f)
            all_models_eval_data[select_model.value] = eval_data

    return pd.DataFrame(
        [
            {
                "model": str(select_model).split(".")[-1],
                **all_models_eval_data[select_model.value][iobb_thrs][0],
            }
            for select_model in models
        ]
    )[["model", "ap", "ar"]]


def plot_iou_result(
    models,
    datasets,
    naming_map,
    disease="all",
    figsize=(10, 10),
    include_recall=False,
    score_thrs=0.05,
):

    cm = plt.get_cmap("rainbow")
    NUM_COLORS = len(models)

    all_models_eval_data = {dataset: {} for dataset in datasets}

    for select_model in models:
        for dataset in datasets:
            with open(
                os.path.join(
                    "eval_results",
                    f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
                ),
                "rb",
            ) as f:
                eval_data = pickle.load(f)
                all_models_eval_data[dataset][select_model.value] = eval_data

    fig, axes = plt.subplots(
        len(datasets),
        2 if include_recall else 1,
        figsize=figsize,
        dpi=120,
        sharex=True,
        squeeze=False,
    )

    for i, dataset in enumerate(datasets):
        axes[i, 0].set_title(f"[{dataset}] - Average Precision")
        axes[i, 0].set_prop_cycle(
            "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        )

        for select_model in models:
            axes[i, 0].plot(
                all_models_eval_data[dataset][select_model.value].keys(),
                [
                    v[0]["ap"]
                    for v in all_models_eval_data[dataset][select_model.value].values()
                ],
                marker="o",
                label=get_model_name(select_model, naming_map=naming_map),
                # color="darkorange",
            )
        axes[i, 0].legend(loc="lower left")
        axes[i, 0].set_xlabel("IoBB threshold")

        if include_recall:

            axes[i, 1].set_title(f"[{dataset}] - Average Recall")
            axes[i, 1].set_prop_cycle(
                "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
            )

            for select_model in models:
                axes[i, 1].plot(
                    all_models_eval_data[dataset][select_model.value].keys(),
                    [
                        v[0]["ar"]
                        for v in all_models_eval_data[dataset][
                            select_model.value
                        ].values()
                    ],
                    marker="o",
                    label=get_model_name(select_model, naming_map=naming_map),
                    # color="darkorange",
                )

            axes[i, 1].legend(loc="lower left")
            axes[i, 1].set_xlabel("IoBB threshold")

    plt.tight_layout()
    plt.plot()
    plt.pause(0.01)

    return fig


def showModelOnDatasets(
    select_model,
    datasets,
    naming_map,
    disease="all",
    figsize=(10, 10),
    include_recall=False,
    score_thrs=0.05,
):
    """
    This function used for detecting the overfitting dataset.
    """
    cm = plt.get_cmap("gist_rainbow")
    NUM_COLORS = len(datasets)

    all_models_eval_data = {}
    for dataset in datasets:
        with open(
            os.path.join(
                "eval_results",
                f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
            ),
            "rb",
        ) as f:
            eval_data = pickle.load(f)
            all_models_eval_data[dataset] = eval_data

    fig, axes = plt.subplots(
        2 if include_recall else 1,
        figsize=figsize,
        dpi=120,
        sharex=True,
        squeeze=False,
    )

    axes = axes[0]

    fig.suptitle(get_model_name(select_model, naming_map=naming_map))

    axes[0].set_title("Average Precision")
    axes[0].set_prop_cycle(
        "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    )

    for dataset in datasets:
        axes[0].plot(
            all_models_eval_data[dataset].keys(),
            [v[0]["ap"] for v in all_models_eval_data[dataset].values()],
            marker="o",
            label=dataset,
            # color="darkorange",
        )
    axes[0].legend(loc="lower left")

    if include_recall:
        axes[1].set_title("Average Recall")
        axes[1].set_prop_cycle(
            "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        )

        for dataset in datasets:
            axes[1].plot(
                all_models_eval_data[dataset].keys(),
                [v[0]["ar"] for v in all_models_eval_data[dataset].values()],
                marker="o",
                label=dataset,
                # color="darkorange",
            )

        axes[1].legend(loc="lower left")
        axes[1].set_xlabel("IoBB")

    plt.tight_layout()
    plt.plot()
    plt.pause(0.01)

    return fig


def showModelsOnDatasets(
    select_models,
    datasets,
    naming_map,
    disease="all",
    figsize=(10, 10),
    score_thrs=0.05,
):
    """
    This function used for detecting the overfitting dataset.
    """
    cm = plt.get_cmap("gist_rainbow")
    NUM_COLORS = len(datasets)

    fig, axes = plt.subplots(
        1,
        len(select_models),
        figsize=figsize,
        dpi=120,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig.suptitle("Average Precision")

    for (
        c_i,
        select_model,
    ) in enumerate(select_models):
        all_models_eval_data = {}
        for dataset in datasets:
            with open(
                os.path.join(
                    "eval_results",
                    f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
                ),
                "rb",
            ) as f:
                eval_data = pickle.load(f)
                all_models_eval_data[dataset] = eval_data

        ax = axes[0][c_i]

        ax.set_title(f"{get_model_name(select_model, naming_map=naming_map)}")
        ax.set_prop_cycle(
            "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        )

        for dataset in datasets:
            ax.plot(
                all_models_eval_data[dataset].keys(),
                [v[0]["ap"] for v in all_models_eval_data[dataset].values()],
                marker="o",
                label=dataset,
                # color="darkorange",
            )
        ax.legend(loc="lower left")
        ax.set_xlabel("IoBB threshold")
    plt.tight_layout()
    plt.plot()
    plt.pause(0.01)

    return fig


def showModelOnScoreThrs(
    select_model,
    dataset: str,
    naming_map,
    disease="all",
    figsize=(10, 10),
    include_recall=False,
    score_thresholds=[0.5, 0.3, 0.2, 0.1, 0.05],
):
    """
    This function used for detecting the overfitting dataset.
    """
    cm = plt.get_cmap("gist_rainbow")
    NUM_COLORS = len(score_thresholds)

    all_models_eval_data = {}
    for score_thrs in score_thresholds:
        with open(
            os.path.join(
                "eval_results",
                f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
            ),
            "rb",
        ) as f:
            eval_data = pickle.load(f)
            all_models_eval_data[score_thrs] = eval_data

    fig, axes = plt.subplots(
        2 if include_recall else 1,
        figsize=figsize,
        dpi=80,
        sharex=True,
        squeeze=False,
    )

    axes = axes[0]

    fig.suptitle(get_model_name(select_model, naming_map=naming_map))

    axes[0].set_title("Average Precision")
    axes[0].set_prop_cycle(
        "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    )

    for score_thrs in score_thresholds:
        axes[0].plot(
            all_models_eval_data[score_thrs].keys(),
            [v[0]["ap"] for v in all_models_eval_data[score_thrs].values()],
            marker="o",
            label=f"score_thrs={str(score_thrs)}",
            # color="darkorange",
        )
    axes[0].legend(loc="lower left")

    if include_recall:
        axes[1].set_title("Average Recall")
        axes[1].set_prop_cycle(
            "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        )

        for score_thrs in score_thresholds:
            axes[1].plot(
                all_models_eval_data[score_thrs].keys(),
                [v[0]["ar"] for v in all_models_eval_data[score_thrs].values()],
                marker="o",
                label=f"score_thrs={str(score_thrs)}",
                # color="darkorange",
            )

        axes[1].legend(loc="lower left")
        axes[1].set_xlabel("IoBB")

    plt.plot()
    plt.pause(0.01)

    return fig


def get_mAP_mAR(
    models,
    datasets: List[str],
    naming_map,
    score_thrs: float = 0.05,
):

    labels_cols = DEFAULT_REFLACX_LABEL_COLS + ["all"]
    # remove the labels that has "/" sign.
    labels_cols = [l.replace("/", "or") for l in labels_cols]
    # score_thrs = 0.05

    all_df = {d: {} for d in labels_cols}

    for disease_str in labels_cols:
        for select_model in models:
            model_path = select_model.value
            eval_df = pd.read_csv(
                os.path.join(
                    "eval_results",
                    f"{model_path}_{disease_str}_score_thrs{score_thrs}.csv",
                ),
                index_col=0,
            )
            all_df[disease_str][model_path] = eval_df

    # eval_dataset = 'val' # ['test', 'val', 'our']

    for eval_dataset in datasets:
        model_dfs = OrderedDict({})

        for select_model in models:
            model_path = select_model.value
            model_name = get_model_name(
                select_model, naming_map=naming_map
            )  # str(select_model).split(".")[-1]
            # Pick dataset

            model_eval = []
            for disease_str in labels_cols:
                model_eval.append(
                    {
                        **dict(
                            all_df[disease_str][model_path][
                                all_df[disease_str][model_path]["dataset"]
                                == eval_dataset
                            ].iloc[0]
                        ),
                        "disease": disease_str,
                    }
                )

            # model_dfs[model_name] = pd.DataFrame(model_eval)[
            #     ["disease", f"AP@[IoBB = 0.50:0.95]", f"AR@[IoBB = 0.50:0.95]"]
            # ]

            model_dfs[model_name] = pd.DataFrame(model_eval)[
                ["disease", f"AP@[IoBB = 0.50]", f"AR@[IoBB = 0.50]"]
            ]

        for idx, k in enumerate(model_dfs.keys()):
            if idx == 0:
                # create the merged df
                merged_df = model_dfs[k].copy()
                merged_df.columns = [
                    "disease" if c == "disease" else f"{c}_{k}"
                    for c in merged_df.columns
                ]
            else:
                df = model_dfs[k].copy()
                df.columns = [
                    "disease" if c == "disease" else f"{c}_{k}" for c in df.columns
                ]
                merged_df = merged_df.merge(
                    df,
                    "left",
                    on="disease",
                )

        print_f.print_title(f"Dataset [{eval_dataset}]")
        display(merged_df)

        merged_df.to_csv(
            os.path.join(f"{eval_dataset}_dataset_class_ap_score_thrs_{score_thrs}.csv")
        )

        return merged_df


def get_performance(
    dataset, all_tasks, evaluator, iouThr=None, areaRng="all", maxDets=30
):
    performance_dict = {}

    if TaskStrs.LESION_DETECTION in all_tasks:
        p_dict = get_ap_ar(
            evaluator[TaskStrs.LESION_DETECTION].coco_eval["bbox"],
            iouThr=iouThr,
            areaRng=areaRng,
            maxDets=maxDets,
        )

        # change to froc
        all_dts = evaluator[TaskStrs.LESION_DETECTION].all_dts
        all_gts = evaluator[TaskStrs.LESION_DETECTION].all_gts

        stats, lls_accuracy, nlls_per_image = get_froc_curve(
            dataset=dataset,
            dts=all_dts,
            all_gts=all_gts,
            plot_title=None,
            use_iou=True,
            n_sample_points=200,
            froc_save_folder="./froc_figures",
        )

        froc_v = get_interpolate_froc(
            stats=stats,
            lls_accuracy=lls_accuracy,
            nlls_per_image=nlls_per_image,
            cat_id=None,
            fps_per_img=[0.5, 1, 2, 4],
            weight=True,
        )

        p_dict["froc"] = np.mean(froc_v)
        performance_dict.update({TaskStrs.LESION_DETECTION: p_dict})

    if TaskStrs.FIXATION_GENERATION in all_tasks:
        performance_dict.update(
            {
                TaskStrs.FIXATION_GENERATION: evaluator[
                    TaskStrs.FIXATION_GENERATION
                ].get_performance_dict()
                # {
                #     "iou": evaluator[TaskStrs.FIXATION_GENERATION].get_iou()
                # }
            }
        )

    if TaskStrs.CHEXPERT_CLASSIFICATION in all_tasks:
        performance_dict.update(
            {
                TaskStrs.CHEXPERT_CLASSIFICATION: evaluator[
                    TaskStrs.CHEXPERT_CLASSIFICATION
                ].get_performance_dict()
                #     {
                #     "auc": evaluator[
                #         TaskStrs.CHEXPERT_CLASSIFICATION
                #     ].get_clf_score(roc_auc_score)
                # }
            }
        )

    if TaskStrs.NEGBIO_CLASSIFICATION in all_tasks:
        performance_dict.update(
            {
                TaskStrs.NEGBIO_CLASSIFICATION: evaluator[
                    TaskStrs.NEGBIO_CLASSIFICATION
                ].get_performance_dict()
                #     {
                #     "auc": evaluator[
                #         TaskStrs.NEGBIO_CLASSIFICATION
                #     ].get_clf_score(roc_auc_score)
                # },
            }
        )

    if TaskStrs.XRAY_CLINICAL_CL in all_tasks:
        performance_dict.update(
            {
                TaskStrs.XRAY_CLINICAL_CL: evaluator[
                    TaskStrs.XRAY_CLINICAL_CL
                ].get_performance_dict()
                #     {
                #     "auc": evaluator[
                #         TaskStrs.CHEXPERT_CLASSIFICATION
                #     ].get_clf_score(roc_auc_score)
                # }
            }
        )

    """
    clinical data
    """
    
    if TaskStrs.AGE_REGRESSION in all_tasks:
        performance_dict.update(
            {
                TaskStrs.AGE_REGRESSION: evaluator[
                    TaskStrs.AGE_REGRESSION
                ].get_performance_dict()
            }
        )
    if TaskStrs.TEMPERATURE_REGRESSION in all_tasks:

        performance_dict.update(
            {
                TaskStrs.TEMPERATURE_REGRESSION: evaluator[
                    TaskStrs.TEMPERATURE_REGRESSION
                ].get_performance_dict()
            }
        )
    if TaskStrs.HEARTRATE_REGRESSION in all_tasks:

        performance_dict.update(
            {
                TaskStrs.HEARTRATE_REGRESSION: evaluator[
                    TaskStrs.HEARTRATE_REGRESSION
                ].get_performance_dict()
            }
        )
    if TaskStrs.RESPRATE_REGRESSION in all_tasks:

        performance_dict.update(
            {
                TaskStrs.RESPRATE_REGRESSION: evaluator[
                    TaskStrs.RESPRATE_REGRESSION
                ].get_performance_dict()
            }
        )
    if TaskStrs.O2SAT_REGRESSION in all_tasks:

        performance_dict.update(
            {
                TaskStrs.O2SAT_REGRESSION: evaluator[
                    TaskStrs.O2SAT_REGRESSION
                ].get_performance_dict()
            }
        )
    if TaskStrs.SBP_REGRESSION in all_tasks:

        performance_dict.update(
            {
                TaskStrs.SBP_REGRESSION: evaluator[
                    TaskStrs.SBP_REGRESSION
                ].get_performance_dict()
            }
        )
    if TaskStrs.DBP_REGRESSION in all_tasks:

        performance_dict.update(
            {
                TaskStrs.DBP_REGRESSION: evaluator[
                    TaskStrs.DBP_REGRESSION
                ].get_performance_dict()
            }
        )
    if TaskStrs.ACUITY_REGRESSION in all_tasks:

        performance_dict.update(
            {
                TaskStrs.ACUITY_REGRESSION: evaluator[
                    TaskStrs.ACUITY_REGRESSION
                ].get_performance_dict()
            }
        )

    if TaskStrs.GENDER_CLASSIFICATION in all_tasks:

        performance_dict.update(
            {
                TaskStrs.GENDER_CLASSIFICATION: evaluator[
                    TaskStrs.GENDER_CLASSIFICATION
                ].get_performance_dict()
            }
        )

    return performance_dict
