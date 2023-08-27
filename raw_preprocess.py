import copy
import os
import json
import pandas as pd
from tqdm import tqdm

TRAIN_DATASETS_TEMPLATES_01 = [
    "TL_span_extraction.json",
    "TL_span_inference.json",
    "TL_text_entailment.json",
    "행정문서1.json",
    "행정문서2.json",
    "행정문서3.json",
    "행정문서4.json",
    "행정문서5.json",
    "TL_unanswerable.json",
    "행정문서_응답불가.json",
    "도서_220419_add.json",
]

TRAIN_DATASETS_TEMPLATES_02 = ["ko_nia_normal_squad_all.json"]
TRAIN_DATASETS_TEMPLATES_03 = ["ko_nia_noanswer_squad_all.json"]

# EVAL_DATASETS_TEMPLATES_01 =


def raw_preprocess(
    templates_list, dir_path, output_list, is_impossible_in_raw: bool = True, all_impossible_value: bool = False
):
    for file_name in templates_list:
        with open(os.path.join(dir_path, file_name), "r", encoding="utf-8") as f:
            sample = json.load(f)
        for each in tqdm(sample["data"], desc=file_name):
            data_dict = dict()
            for paragraph in each["paragraphs"]:
                data_dict["context"] = paragraph["context"]
                for qa in paragraph["qas"]:
                    data_dict["question"] = qa["question"]
                    if is_impossible_in_raw:
                        data_dict["is_impossible"] = int(qa["is_impossible"])
                    else:
                        data_dict["is_impossible"] = int(all_impossible_value)
                    output_list.append(copy.deepcopy(data_dict))


def pd_to_list_dict(total_list):
    pd_total = pd.DataFrame.from_records(total_list)
    min_value = pd_total.groupby("is_impossible").count()["question"].min()
    true_sampling = pd_total[pd_total["is_impossible"] == True].sample(n=min_value)
    false_sampling = pd_total[pd_total["is_impossible"] == False].sample(n=min_value)
    total_sampling = pd.concat([true_sampling, false_sampling])

    return total_sampling.to_dict("records")


print("@@@@@@@@@@ train raw preprocess @@@@@@@@@@")
train_dir = "./data/train"
train_output = "train_data.json"
if not os.path.exists(os.path.join(train_dir, train_output)):
    train_list = []

    raw_preprocess(TRAIN_DATASETS_TEMPLATES_01, train_dir, train_list, is_impossible_in_raw=True)
    raw_preprocess(
        TRAIN_DATASETS_TEMPLATES_02, train_dir, train_list, is_impossible_in_raw=False, all_impossible_value=False
    )
    raw_preprocess(
        TRAIN_DATASETS_TEMPLATES_03, train_dir, train_list, is_impossible_in_raw=False, all_impossible_value=True
    )
    dict_data = pd_to_list_dict(train_list)
    with open(os.path.join(train_dir, train_output), "w", encoding="utf-8") as file:
        file.write(json.dumps(dict_data, indent=4))

VALID_DATASETS_TEMPLATES_01 = [
    "VL_span_extraction.json",
    "VL_span_inference.json",
    "VL_text_entailment.json",
    "행정문서1.json",
    "행정문서2.json",
    "행정문서3.json",
    "행정문서4.json",
    "행정문서5.json",
    "VL_unanswerable.json",
    "행정문서_응답불가.json",
    "도서.json",
]

print("@@@@@@@@@@ validation raw preprocess @@@@@@@@@@")
valid_dir = "./data/validation"
valid_output = "valid_data.json"
if not os.path.exists(os.path.join(valid_dir, valid_output)):
    valid_list = []

    raw_preprocess(VALID_DATASETS_TEMPLATES_01, valid_dir, valid_list, is_impossible_in_raw=True)
    dict_data = pd_to_list_dict(valid_list)
    with open(os.path.join(valid_dir, valid_output), "w", encoding="utf-8") as file:
        file.write(json.dumps(dict_data, indent=4))
