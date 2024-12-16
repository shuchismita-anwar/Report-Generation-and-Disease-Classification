from collections import OrderedDict
import os
import random
from sklearn.preprocessing import LabelEncoder
import torch
import json

import pandas as pd
import numpy as np
import torch.utils.data as data
import pandas as pd
import numpy as np
import torch.utils.data as data
import sys
import torch.nn.functional as F
from typing import Callable, Dict, List, Tuple, Union
from pathlib import Path
from PIL import Image
from copy import deepcopy
from data.strs import SourceStrs, TaskStrs
from data.utils import chain_map
from torchvision import transforms
from models.setup import ModelSetup
from .constants import (
    DEFAULT_REFLACX_BOX_COORD_COLS,
    DEFAULT_REFLACX_BOX_FIX_COLS,
    REFLACX_ALL_LABEL_COLS,
    DEFAULT_REFLACX_LABEL_COLS,
    DEFAULT_REFLACX_PATH_COLS,
    DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
)
from .paths import SPREADSHEET_FOLDER


from .helpers import map_dict_to_device, map_target_to_device, target_processing
from .fixation import get_fixations_dict_from_fixation_df, get_heatmap

from torchvision.transforms import functional as tvF
from sklearn.preprocessing import normalize


def collate_fn(batch: Tuple) -> Tuple:
    return tuple(zip(*batch))


training_clinical_mean_std = {
    "age": {"mean": 62.924050632911396, "std": 18.486667896662354},
    "temperature": {"mean": 98.08447784810126, "std": 2.7465209372955712},
    "heartrate": {"mean": 85.95379746835444, "std": 18.967507646992733},
    "resprate": {"mean": 18.15221518987342, "std": 2.6219004903965004},
    "o2sat": {"mean": 97.85411392405064, "std": 2.6025150031174946},
    "sbp": {"mean": 133.0685126582279, "std": 25.523304795054102},
    "dbp": {"mean": 74.01107594936708, "std": 16.401336318103716},
    "acuity": {"mean": 2.2610759493670884, "std": 0.7045539799670345},
}


class PhysioNetClincalDataset(data.Dataset):
    def __init__(
        self,
        PHYSIONET_PATH: str,
        clinical_numerical_cols,
        clinical_categorical_cols,
        normalise_clinical_num=False,
        # bbox_to_mask=True,
        split_str: str = None,
        # dataset_mode: str = "normal",
        # labels_cols: List[str] = DEFAULT_REFLACX_LABEL_COLS,
        # all_disease_cols: List[str] = REFLACX_ALL_LABEL_COLS,
        # repetitive_label_map: Dict[
        #     str, List[str]
        # ] = DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
        # box_fix_cols: List[str] = DEFAULT_REFLACX_BOX_FIX_COLS,
        # box_coord_cols: List[str] = DEFAULT_REFLACX_BOX_COORD_COLS,
        path_cols: List[str] = ["image_path"],
        spreadsheets_folder=SPREADSHEET_FOLDER,
        ### input & label fields ###
        with_xrays_input: bool = True,
        with_clincal_input: bool = True,
        with_clinical_label: bool = False,
        # with_bboxes_label: bool = True,
        # with_fixations_label: bool = True,
        # with_fixations_input: bool = True,
        # fixations_mode_label="reporting",  # [silent, reporting, all]
        # fixations_mode_input="reporting",
        with_chexpert_label: bool = True,
        with_negbio_label: bool = True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        image_size=512,
        random_flip=True,
        use_aug=False,
        # use_clinical_df=False,
    ):
        # Data loading selections

        self.split_str: str = split_str
        self.random_flip = random_flip
        # image related params.
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.use_aug = self.split_str == 'train' and use_aug

        # self.bbox_to_mask = bbox_to_mask
        # self.use_clinical_df = use_clinical_df
        self.PHYSIONET_PATH = PHYSIONET_PATH
        self.path_cols = path_cols
        # self.path_cols: List[str] = path_cols

        # Labels
        # self.labels_cols: List[str] = labels_cols
        # self.all_disease_cols: List[str] = all_disease_cols
        # self.repetitive_label_map: Dict[str, List[str]] = repetitive_label_map
        # self.dataset_mode: str = dataset_mode
        self.with_clinical_label = with_clinical_label

        self.with_clinical_input = with_clincal_input
        # self.should_load_clinical_df = self.with_clinical_input or self.with_clinical_label or self.use_clinical_df
        # if self.should_load_clinical_df:
        self.clinical_numerical_cols = clinical_numerical_cols
        self.normalise_clinical_num = normalise_clinical_num
        self.clinical_categorical_cols = clinical_categorical_cols
        self.with_xrays_input = with_xrays_input

        self.with_chexpert_label = with_chexpert_label
        self.with_negbi_label = with_negbio_label
        # self.with_bboxes_label = with_bboxes_label
        # if self.with_bboxes_label:
        #     self.box_fix_cols: List[str] = box_fix_cols
        #     self.box_coord_cols: List[str] = box_coord_cols

        # self.with_fixations_label: bool = with_fixations_label
        # if self.with_fixations_label:
        #     self.fiaxtions_mode_label = fixations_mode_label

        # self.with_fixations_input = with_fixations_input
        # if self.with_fixations_input:
        #     self.fiaxtions_mode_input = fixations_mode_input

        # deciding which to df load
        # self.use_discrete_clinical = use_discrete_clinical
        self.df_path = "physio_clinical_cxr_meta.csv"

        # raise StopIteration(f"The dataset now loading is {self.df_path}")
        self.df: pd.DataFrame = pd.read_csv(
            os.path.join(spreadsheets_folder, self.df_path), index_col=0
        )

        # get the splited group that we desire.
        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        self.replace_paths()

        # preprocessing data.
        # self.preprocess_label()

        self.chexpert_label_cols = [
            "Atelectasis_chexpert",
            "Cardiomegaly_chexpert",
            "Consolidation_chexpert",
            "Edema_chexpert",
            "Enlarged Cardiomediastinum_chexpert",
            "Fracture_chexpert",
            "Lung Lesion_chexpert",
            "Lung Opacity_chexpert",
            "No Finding_chexpert",
            "Pleural Effusion_chexpert",
            "Pleural Other_chexpert",
            "Pneumonia_chexpert",
            "Pneumothorax_chexpert",
            "Support Devices_chexpert",
        ]
        self.negbio_label_cols = [
            "Atelectasis_negbio",
            "Cardiomegaly_negbio",
            "Consolidation_negbio",
            "Edema_negbio",
            "Enlarged Cardiomediastinum_negbio",
            "Fracture_negbio",
            "Lung Lesion_negbio",
            "Lung Opacity_negbio",
            "No Finding_negbio",
            "Pleural Effusion_negbio",
            "Pleural Other_negbio",
            "Pneumonia_negbio",
            "Pneumothorax_negbio",
            "Support Devices_negbio",
        ]

        # remove all numerical cols:
        # In case we used it somewhere.

        # cols_to_remove = [
        #     "age",
        #     "temperature",
        #     "heartrate",
        #     "resprate",
        #     "o2sat",
        #     "sbp",
        #     "dbp",
        #     # "pain",
        #     "acuity",
        # ]

        # for col in cols_to_remove:
        #     del self.df[col]

        ## check if the col is still df:
        # for col in cols_to_remove:
        #     if col in self.df.columns:
        #         raise StopIteration(f"Column {col} is not removed from dataframe.")

        if self.with_clinical_input or self.with_clinical_label:
            self.preprocess_clinical_df()

        ### create aug transformer here.

        self.resize_transform = Flip_resize_transform = transforms.Compose(
            [
                transforms.Resize([image_size, image_size]),
            ]
        )

        self.aug_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomResizedCrop([image_size, image_size], scale=(0.2, 1.0)),
            ]
        )

        super().__init__()

    # def preprocess_label(
    #     self,
    # ):
    #     self.df[self.all_disease_cols] = self.df[self.all_disease_cols].gt(0)

    def replace_paths(
        self,
    ):
        # replace the path with local mimic folder path.
        replace_str = "{PHYSIONET_PATH}"
        for p_col in self.path_cols:
            if p_col in self.df.columns:
                apply_fn: Callable[[str], str] = lambda x: str(
                    Path(x.replace(replace_str, self.PHYSIONET_PATH))
                )
                self.df[p_col] = self.df[p_col].apply(apply_fn)

    def load_image_array(self, image_path: str) -> np.ndarray:
        return np.asarray(Image.open(image_path))

    def plot_image_from_array(self, image_array: np.ndarray):
        im = Image.fromarray(image_array)
        im.show()

    def negbio_chexpert_disease_to_idx(disease, label_cols):
        if not disease in label_cols:
            raise Exception("This disease is not the label.")

        return label_cols.index(disease)

    def negbio_chexpert_idx_to_disease(idx, label_cols):
        if idx >= len(label_cols):
            return f"exceed label range :{idx}"

        return label_cols[idx]


    def __len__(self) -> int:
        return len(self.df)

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize_image(self, image: torch.Tensor, size):

        image = torch.nn.functional.interpolate(
            image[None],
            size=size,
            mode="bilinear",
            align_corners=False,
        )[0]

        return image

    def prepare_xray(self, xray):
        if self.use_aug and random.random() <0.95: # 5% for direct resize.
            xray = self.aug_transform(xray)
            # print(f"Using Augmentation in {self.split_str}.")
            if self.split_str == 'test' or self.split_str == 'val':
                raise StopIteration(f"Shouldn't use Augmentation in {self.split_str}")
        else:
            xray = self.resize_transform(xray)

        xray = self.normalize(xray)
        # xray = self.resize_image(image=xray, size=[self.image_size, self.image_size])
        return xray

    def prepare_clinical_labels(self, data):
        clinical_labels_dict = {}

        for c in self.clinical_numerical_cols + self.clinical_categorical_cols:
            clinical_labels_dict.update({c: data[c]})

        return clinical_labels_dict

    def prepare_clinical(self, data):
        clinical_num = None
        if (
            not self.clinical_numerical_cols is None
            and len(self.clinical_numerical_cols) > 0
        ):
            # if self.normalise_clinical_num:
            #     clinical_num = (
            #         torch.tensor(
            #             self.clinical_num_norm.transform(
            #                 np.array([data[self.clinical_numerical_cols]])
            #             ),
            #             dtype=float,
            #         )
            #         .float()
            #         .squeeze()
            #     )
            # else:
            clinical_num = torch.tensor(
                np.array(data[self.clinical_numerical_cols], dtype=float)
            ).float()

        clinical_cat = None
        if (
            not self.clinical_categorical_cols is None
            and len(self.clinical_categorical_cols) > 0
        ):
            clinical_cat = {
                c: torch.tensor(np.array(data[c], dtype=int))
                for c in self.clinical_categorical_cols
            }

        return clinical_cat, clinical_num


    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        # find the df
        data: pd.Series = self.df.iloc[idx]

        # it's necesary to load the image, becasue it will be used to run the transform.
        xray: Image = Image.open(data["image_path"]).convert("RGB")
        xray_height, xray_width = xray.height, xray.width
        xray = tvF.to_tensor(xray)
        original_image_size = xray.shape[-2:]

        # flip = self.random_flip and random.random() < 0.5
        # contain this one into thhe

        """
        inputs
        """
        input_dict = OrderedDict({})

        if self.with_xrays_input:
            xray = self.prepare_xray(xray)
            input_dict.update({SourceStrs.XRAYS: {"images": xray}})

        if self.with_clinical_input:
            clinical_cat, clinical_num = self.prepare_clinical(data)
            input_dict.update(
                {SourceStrs.CLINICAL: {"cat": clinical_cat, "num": clinical_num}}
            )

        target = OrderedDict({})
  
        if self.with_clinical_label:
            # clinical_label_dict = self.prepare_clinical_labels(data)
            target.update(
                {TaskStrs.AGE_REGRESSION: {"regressions": torch.tensor(data["age"])}}
            )

            target.update(
                {
                    TaskStrs.TEMPERATURE_REGRESSION: {
                        "regressions": torch.tensor(data["temperature"])
                    }
                }
            )

            target.update(
                {
                    TaskStrs.HEARTRATE_REGRESSION: {
                        "regressions": torch.tensor(data["heartrate"])
                    }
                }
            )

            target.update(
                {
                    TaskStrs.RESPRATE_REGRESSION: {
                        "regressions": torch.tensor(data["resprate"])
                    }
                }
            )

            target.update(
                {
                    TaskStrs.O2SAT_REGRESSION: {
                        "regressions": torch.tensor(data["o2sat"])
                    }
                }
            )

            target.update(
                {TaskStrs.SBP_REGRESSION: {"regressions": torch.tensor(data["sbp"])}}
            )

            target.update(
                {TaskStrs.DBP_REGRESSION: {"regressions": torch.tensor(data["dbp"])}}
            )

            target.update(
                {
                    TaskStrs.ACUITY_REGRESSION: {
                        "regressions": torch.tensor(data["acuity"])
                    }
                }
            )

            target.update(
                {
                    TaskStrs.GENDER_CLASSIFICATION: {
                        "classifications": torch.tensor(data["gender"]).unsqueeze(0)
                    }
                }
            )

        if self.with_chexpert_label:
            target.update(
                {
                    TaskStrs.CHEXPERT_CLASSIFICATION: {
                        "classifications": torch.tensor(data[self.chexpert_label_cols])
                        == 1
                    }
                }
            )

        if self.with_negbi_label:
            target.update(
                {
                    TaskStrs.NEGBIO_CLASSIFICATION: {
                        "classifications": torch.tensor(data[self.negbio_label_cols])
                        == 1
                    }
                }
            )

        # if self.with_fixations_label:
        #     fix = self.prepare_fixation(data, self.fiaxtions_mode_label)
        #     if flip:
        #         fix = fix.flip(-1)

        #     target.update({TaskStrs.FIXATION_GENERATION: {"heatmaps": fix}})

        # img_t, target = self.transforms(xray, target)

        # if self.with_xrays:
        # input_dict.update({SourceStrs.XRAYS: {"images": img_t}})

        ## we should perform the preprocessing in here instead of using that transformer in the task_perfromer.
        # record the original size of the image

        return input_dict, target

    def prepare_input_from_data(
        self,
        data: Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
            Tuple[torch.Tensor, Dict],
        ],
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        inputs, targets = data

        inputs = list(inputs)
        targets = list(targets)

        return inputs, targets

    def get_idxs_from_dicom_id(self, dicom_id: str) -> List[str]:
        return [
            self.df.index.get_loc(i)
            for i in self.df.index[self.df["dicom_id"].eq(dicom_id)]
        ]

    def get_image_path_from_dicom_id(self, dicom_id: str) -> List[str]:
        return self.df[self.df["dicom_id"] == dicom_id].iloc[0]["image_path"]

    def preprocess_clinical_df(
        self,
    ):
        self.encoders_map: Dict[str, LabelEncoder] = {}

        # encode the categorical cols.
        for col in self.clinical_categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders_map[col] = le

        if self.normalise_clinical_num:
            self.clinical_std_mean = {}

            for col in self.clinical_numerical_cols:
                # calculate mean and std
                mean = training_clinical_mean_std[col]["mean"]
                std = training_clinical_mean_std[col]["std"]
                self.df[col] = (self.df[col] - mean) / std
                # self.df[col] = normalize([self.df[col]], axis=1)[0]


class ReflacxDataset(data.Dataset):
    """
    Class to load the preprocessed REFLACX master sheet. There `.csv` files are required to run this class.

    - `reflacx_for_eyetracking.csv'

    """

    def __init__(
        self,
        MIMIC_EYE_PATH: str,
        clinical_numerical_cols,
        clinical_categorical_cols,
        normalise_clinical_num=False,
        bbox_to_mask=True,
        split_str: str = None,
        dataset_mode: str = "normal",
        labels_cols: List[str] = DEFAULT_REFLACX_LABEL_COLS,
        all_disease_cols: List[str] = REFLACX_ALL_LABEL_COLS,
        repetitive_label_map: Dict[
            str, List[str]
        ] = DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
        box_fix_cols: List[str] = DEFAULT_REFLACX_BOX_FIX_COLS,
        box_coord_cols: List[str] = DEFAULT_REFLACX_BOX_COORD_COLS,
        path_cols: List[str] = DEFAULT_REFLACX_PATH_COLS,
        spreadsheets_folder=SPREADSHEET_FOLDER,
        ### input & label fields ###
        with_xrays_input: bool = True,
        with_clincal_input: bool = True,
        with_clinical_label: bool = False,
        with_bboxes_label: bool = True,
        with_fixations_label: bool = True,
        with_fixations_input: bool = True,
        fixations_mode_label="reporting",  # [silent, reporting, all]
        fixations_mode_input="reporting",
        with_chexpert_label: bool = True,
        with_negbio_label: bool = True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        image_size=512,
        random_flip=True,
        use_clinical_df=False,
    ):
        # Data loading selections

        self.split_str: str = split_str
        self.random_flip = random_flip
        # image related params.
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std

        self.bbox_to_mask = bbox_to_mask
        self.use_clinical_df = use_clinical_df
        self.MIMIC_EYE_PATH = MIMIC_EYE_PATH
        self.path_cols = path_cols
        self.path_cols: List[str] = path_cols

        # Labels
        self.labels_cols: List[str] = labels_cols
        self.all_disease_cols: List[str] = all_disease_cols
        self.repetitive_label_map: Dict[str, List[str]] = repetitive_label_map
        self.dataset_mode: str = dataset_mode
        self.with_clinical_label = with_clinical_label

        self.with_clinical_input = with_clincal_input
        self.should_load_clinical_df = (
            self.with_clinical_input or self.with_clinical_label or self.use_clinical_df
        )
        if self.should_load_clinical_df:
            self.clinical_numerical_cols = clinical_numerical_cols
            self.normalise_clinical_num = normalise_clinical_num
            self.clinical_categorical_cols = clinical_categorical_cols

        self.with_xrays_input = with_xrays_input

        self.with_chexpert_label = with_chexpert_label
        self.with_negbi_label = with_negbio_label
        self.with_bboxes_label = with_bboxes_label
        if self.with_bboxes_label:
            self.box_fix_cols: List[str] = box_fix_cols
            self.box_coord_cols: List[str] = box_coord_cols

        self.with_fixations_label: bool = with_fixations_label
        if self.with_fixations_label:
            self.fiaxtions_mode_label = fixations_mode_label

        self.with_fixations_input = with_fixations_input
        if self.with_fixations_input:
            self.fiaxtions_mode_input = fixations_mode_input

        # deciding which to df load
        self.df_path = (
            "reflacx_clinical_eye.csv"
            if (self.should_load_clinical_df)
            else "reflacx_eye.csv"
        )

        # raise StopIteration(f"The dataset now loading is {self.df_path}")
        self.df: pd.DataFrame = pd.read_csv(
            os.path.join(spreadsheets_folder, self.df_path), index_col=0
        )

        # get the splited group that we desire.
        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        self.replace_paths()

        # preprocessing data.
        self.preprocess_label()

        self.chexpert_label_cols = [
            c for c in self.df.columns if c.endswith("_chexpert")
        ]
        self.negbio_label_cols = [c for c in self.df.columns if c.endswith("_negbio")]

        if self.with_clinical_input or self.with_clinical_label:
            self.preprocess_clinical_df()

        super().__init__()

    def preprocess_label(
        self,
    ):
        self.df[self.all_disease_cols] = self.df[self.all_disease_cols].gt(0)

    def replace_paths(
        self,
    ):
        # replace the path with local mimic folder path.
        for p_col in self.path_cols:
            if p_col in self.df.columns:
                if p_col == "bbox_paths":

                    def apply_bbox_paths_transform(input_paths_str: str) -> List[str]:
                        input_paths_list: List[str] = json.loads(input_paths_str)
                        replaced_path_list: List[str] = [
                            p.replace("{XAMI_MIMIC_PATH}", self.MIMIC_EYE_PATH)
                            for p in input_paths_list
                        ]
                        return replaced_path_list

                    apply_fn: Callable[
                        [str], List[str]
                    ] = lambda x: apply_bbox_paths_transform(x)

                else:
                    apply_fn: Callable[[str], str] = lambda x: str(
                        Path(x.replace("{XAMI_MIMIC_PATH}", self.MIMIC_EYE_PATH))
                    )

                self.df[p_col] = self.df[p_col].apply(apply_fn)

    def load_image_array(self, image_path: str) -> np.ndarray:
        return np.asarray(Image.open(image_path))

    def plot_image_from_array(self, image_array: np.ndarray):
        im = Image.fromarray(image_array)
        im.show()

    def negbio_chexpert_disease_to_idx(disease, label_cols):
        if not disease in label_cols:
            raise Exception("This disease is not the label.")

        return label_cols.index(disease)

    def negbio_chexpert_idx_to_disease(idx, label_cols):
        if idx >= len(label_cols):
            return f"exceed label range :{idx}"

        return label_cols[idx]

    def disease_to_idx(self, disease: str) -> int:
        if not disease in self.labels_cols:
            raise Exception("This disease is not the label.")

        if disease == "background":
            return 0

        return self.labels_cols.index(disease) + 1

    def label_idx_to_disease(self, idx: int) -> str:
        if idx == 0:
            return "background"

        if idx > len(self.labels_cols):
            return f"exceed label range :{idx}"

        return self.labels_cols[idx - 1]

    def __len__(self) -> int:
        return len(self.df)

    def generate_bboxes_df(
        self,
        ellipse_df: pd.DataFrame,
    ) -> pd.DataFrame:
        boxes_df = ellipse_df[self.box_fix_cols]

        # relabel repetitive columns.
        for k in self.repetitive_label_map.keys():
            boxes_df.loc[:, k] = ellipse_df[
                [l for l in self.repetitive_label_map[k] if l in ellipse_df.columns]
            ].any(axis=1)

        # filtering out the diseases not in the label_cols
        boxes_df = boxes_df[boxes_df[self.labels_cols].any(axis=1)]
        label_df = boxes_df.loc[:, DEFAULT_REFLACX_LABEL_COLS].reset_index(drop=True)

        labels = [
            list(label_df.loc[i, label_df.any()].index) for i in range(len(label_df))
        ]

        boxes_df["label"] = labels

        new_df_list = []

        if len(boxes_df) > 0:
            for _, instance in boxes_df.iterrows():
                for l in instance["label"]:
                    new_df_list.append(
                        {
                            "xmin": instance["xmin"],
                            "ymin": instance["ymin"],
                            "xmax": instance["xmax"],
                            "ymax": instance["ymax"],
                            "label": l,
                        }
                    )

        return pd.DataFrame(
            new_df_list, columns=["xmin", "ymin", "xmax", "ymax", "label"]
        )

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize_image(self, image: torch.Tensor, size):

        image = torch.nn.functional.interpolate(
            image[None],
            size=size,
            mode="bilinear",
            align_corners=False,
        )[0]

        return image

    def prepare_xray(self, xray):
        xray = self.normalize(xray)
        xray = self.resize_image(image=xray, size=[self.image_size, self.image_size])
        return xray

    def prepare_clinical_labels(self, data):
        clinical_labels_dict = {}

        for c in self.clinical_numerical_cols + self.clinical_categorical_cols:
            clinical_labels_dict.update({c: data[c]})

        return clinical_labels_dict

    def prepare_clinical(self, data):
        clinical_num = None
        if (
            not self.clinical_numerical_cols is None
            and len(self.clinical_numerical_cols) > 0
        ):
            # if self.normalise_clinical_num:
            #     clinical_num = (
            #         torch.tensor(
            #             self.clinical_num_norm.transform(
            #                 np.array([data[self.clinical_numerical_cols]])
            #             ),
            #             dtype=float,
            #         )
            #         .float()
            #         .squeeze()
            #     )
            # else:
            clinical_num = torch.tensor(
                np.array(data[self.clinical_numerical_cols], dtype=float)
            ).float()

        clinical_cat = None
        if (
            not self.clinical_categorical_cols is None
            and len(self.clinical_categorical_cols) > 0
        ):
            clinical_cat = {
                c: torch.tensor(np.array(data[c], dtype=int))
                for c in self.clinical_categorical_cols
            }

        return clinical_cat, clinical_num

    def get_fixation_image(self, data, mode):

        fiaxtion_df = pd.read_csv(data["fixation_path"])

        if mode != "normal":
            utterance_path = os.path.join(
                os.path.dirname(data["fixation_path"]),
                "timestamps_transcription.csv",
            )
            utterance_df = pd.read_csv(utterance_path)
            report_starting_time = utterance_df.iloc[0]["timestamp_start_word"]
            if mode == "reporting":
                fiaxtion_df = fiaxtion_df[
                    fiaxtion_df["timestamp_start_fixation"] >= report_starting_time
                ]
            elif mode == "silent":
                fiaxtion_df = fiaxtion_df[
                    fiaxtion_df["timestamp_start_fixation"] < report_starting_time
                ]
            else:
                raise ValueError("Not supported fiaxtions mode.")

        fix = get_heatmap(
            get_fixations_dict_from_fixation_df(fiaxtion_df),
            (data["image_size_x"], data["image_size_y"]),
        ).astype(np.float32)

        return fix

    def prepare_fixation(self, data, mode):
        fix = self.get_fixation_image(data, mode)
        fix = tvF.to_tensor(fix)
        fix = self.normalize(fix)
        fix = self.resize_image(image=fix, size=[self.image_size, self.image_size])
        return fix

    def resize_boxes(
        self, boxes: torch.Tensor, original_size: List[int], new_size: List[int]
    ) -> torch.Tensor:
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device)
            / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def get_lesion_detection_labels(self, idx, data, original_size, new_size):
        bboxes_df = self.generate_bboxes_df(pd.read_csv(data["bbox_path"]))
        bboxes = np.array(bboxes_df[self.box_coord_cols], dtype=float)
        # x1, y1, x2, y2
        unsized_boxes = bboxes
        bboxes = torch.tensor(bboxes)
        bboxes = self.resize_boxes(
            boxes=bboxes, original_size=original_size, new_size=new_size
        )
        # resize the bb
        # Calculate area of boxes.
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        unsized_area = (unsized_boxes[:, 3] - unsized_boxes[:, 1]) * (
            unsized_boxes[:, 2] - unsized_boxes[:, 0]
        )
        labels = torch.tensor(
            np.array(bboxes_df["label"].apply(lambda l: self.disease_to_idx(l))).astype(
                int
            ),
            dtype=torch.int64,
        )

        image_id = torch.tensor([idx])
        num_objs = bboxes.shape[0]
        # S suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # prepare lesion-detection targets
        return {
            "boxes": bboxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "dicom_id": data["dicom_id"],
            "image_path": data["image_path"],
            "original_image_sizes": original_size,
            "unsized_boxes": unsized_boxes,
            "unsized_area": unsized_area,
        }

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        # find the df
        data: pd.Series = self.df.iloc[idx]

        # it's necesary to load the image, becasue it will be used to run the transform.
        xray: Image = Image.open(data["image_path"]).convert("RGB")
        xray_height, xray_width = xray.height, xray.width
        xray = tvF.to_tensor(xray)
        original_image_size = xray.shape[-2:]

        flip = self.random_flip and random.random() < 0.5
        # contain this one into thhe

        """
        inputs
        """
        input_dict = OrderedDict({})

        if self.with_xrays_input:
            xray = self.prepare_xray(xray)
            if flip:
                xray = xray.flip(-1)
            input_dict.update({SourceStrs.XRAYS: {"images": xray}})

        if self.with_clinical_input:
            clinical_cat, clinical_num = self.prepare_clinical(data)
            input_dict.update(
                {SourceStrs.CLINICAL: {"cat": clinical_cat, "num": clinical_num}}
            )

        if self.with_fixations_input:
            fix = self.prepare_fixation(data, self.fiaxtions_mode_input)
            if flip:
                fix = fix.flip(-1)

            input_dict.update({SourceStrs.FIXATIONS: {"images": fix}})

        # do bboxes resizing later.
        # if self.obj_det_task_name in target_index:
        #     bbox = target_index[self.obj_det_task_name]["boxes"]
        #     bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        #     target_index[self.obj_det_task_name]["boxes"] = bbox

        """
        targets
        """

        target = OrderedDict({})

        if self.with_bboxes_label:
            lesion_target = self.get_lesion_detection_labels(
                idx=idx,
                data=data,
                original_size=original_image_size,
                new_size=[self.image_size, self.image_size],
            )

            if flip:
                width = self.image_size
                bbox = lesion_target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                lesion_target["boxes"] = bbox

            if self.bbox_to_mask:
                # generate masks from bboxes
                num_objs = lesion_target["boxes"].shape[0]
                masks = torch.zeros(
                    (num_objs, xray_height, xray_width), dtype=torch.uint8
                )
                for i, b in enumerate(lesion_target["boxes"]):
                    b = b.int()
                    masks[i, b[1] : b[3], b[0] : b[2]] = 1
                lesion_target["masks"] = masks

            target.update({TaskStrs.LESION_DETECTION: lesion_target})

        if self.with_clinical_label:
            # clinical_label_dict = self.prepare_clinical_labels(data)
            target.update(
                {TaskStrs.AGE_REGRESSION: {"regressions": torch.tensor(data["age"])}}
            )

            target.update(
                {
                    TaskStrs.TEMPERATURE_REGRESSION: {
                        "regressions": torch.tensor(data["temperature"])
                    }
                }
            )

            target.update(
                {
                    TaskStrs.HEARTRATE_REGRESSION: {
                        "regressions": torch.tensor(data["heartrate"])
                    }
                }
            )

            target.update(
                {
                    TaskStrs.RESPRATE_REGRESSION: {
                        "regressions": torch.tensor(data["resprate"])
                    }
                }
            )

            target.update(
                {
                    TaskStrs.O2SAT_REGRESSION: {
                        "regressions": torch.tensor(data["o2sat"])
                    }
                }
            )

            target.update(
                {TaskStrs.SBP_REGRESSION: {"regressions": torch.tensor(data["sbp"])}}
            )

            target.update(
                {TaskStrs.DBP_REGRESSION: {"regressions": torch.tensor(data["dbp"])}}
            )

            target.update(
                {
                    TaskStrs.ACUITY_REGRESSION: {
                        "regressions": torch.tensor(data["acuity"])
                    }
                }
            )

            target.update(
                {
                    TaskStrs.GENDER_CLASSIFICATION: {
                        "classifications": torch.tensor(data["gender"]).unsqueeze(0)
                    }
                }
            )

        if self.with_chexpert_label:
            target.update(
                {
                    TaskStrs.CHEXPERT_CLASSIFICATION: {
                        "classifications": torch.tensor(data[self.chexpert_label_cols])
                        == 1
                    }
                }
            )

        if self.with_negbi_label:
            target.update(
                {
                    TaskStrs.NEGBIO_CLASSIFICATION: {
                        "classifications": torch.tensor(data[self.negbio_label_cols])
                        == 1
                    }
                }
            )

        if self.with_fixations_label:
            fix = self.prepare_fixation(data, self.fiaxtions_mode_label)
            if flip:
                fix = fix.flip(-1)

            target.update({TaskStrs.FIXATION_GENERATION: {"heatmaps": fix}})

        # img_t, target = self.transforms(xray, target)

        # if self.with_xrays:
        # input_dict.update({SourceStrs.XRAYS: {"images": img_t}})

        ## we should perform the preprocessing in here instead of using that transformer in the task_perfromer.
        # record the original size of the image

        return input_dict, target

    def prepare_input_from_data(
        self,
        data: Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
            Tuple[torch.Tensor, Dict],
        ],
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        inputs, targets = data

        inputs = list(inputs)
        targets = list(targets)

        return inputs, targets

    def get_idxs_from_dicom_id(self, dicom_id: str) -> List[str]:
        return [
            self.df.index.get_loc(i)
            for i in self.df.index[self.df["dicom_id"].eq(dicom_id)]
        ]

    def get_image_path_from_dicom_id(self, dicom_id: str) -> List[str]:
        return self.df[self.df["dicom_id"] == dicom_id].iloc[0]["image_path"]

    def preprocess_clinical_df(
        self,
    ):
        self.encoders_map: Dict[str, LabelEncoder] = {}

        # encode the categorical cols.
        for col in self.clinical_categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders_map[col] = le

        if self.normalise_clinical_num:
            self.clinical_std_mean = {}

            for col in self.clinical_numerical_cols:
                # calculate mean and std
                mean = training_clinical_mean_std[col]["mean"]
                std = training_clinical_mean_std[col]["std"]
                self.df[col] = (self.df[col] - mean) / std
                # self.df[col] = normalize([self.df[col]], axis=1)[0]
