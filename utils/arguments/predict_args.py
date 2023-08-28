from transformers import TrainingArguments
from dataclasses import field, dataclass
from typing import Literal


@dataclass
class PredictArguments(TrainingArguments):
    model_name_or_path: str = field(
        metadata={"help": "predict에 사용할 모델의 이름 혹은 폴더 경로"},
    )
    data_file_path: str = field(
        metadata={"help": "predict을 수행할 데이터 파일의 경로"},
    )
    data_fils_ext: Literal["json", "csv", "parquet", "text"] = field(
        default="json",
        metadata={"help": "불러들일 파일의 확장자"},
    )
    save_dir: str = field(
        default="predict의 결과물을 저장할 폴더의 경로",
        metadata={"help": ""},
    )
    output_dir: str = field(
        default="./",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
