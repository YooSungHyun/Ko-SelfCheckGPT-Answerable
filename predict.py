# for type annotation
import os
from dataclasses import dataclass, field
from typing import Dict, Literal

import numpy as np
import torch
from datasets import Dataset, load_dataset
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction
from utils import clean_unicode, trim_text

PASSAGE_QUESTION_TOKEN = "[SEP]"

os.environ["TORCHDYNAMO_DISABLE"] = "1"


class PredictArguments(TrainingArguments):
    model_name_or_path: str = field(
        default=None,
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
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


def main(parser: HfArgumentParser) -> None:
    predict_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    set_seed(predict_args.seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        predict_args.model_name_or_path,
        attention_type="original_full",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(predict_args.model_name_or_path, use_fast=True)

    raw_test_datasets = load_dataset(
        predict_args.data_fils_ext,
        data_files=predict_args.data_file_path,
        split="all",
    )

    def preprocess(raw):
        input_text = (
            clean_unicode(trim_text(raw["context"]))
            + PASSAGE_QUESTION_TOKEN
            + clean_unicode(trim_text(raw["question"]))
        )
        tokenized_text = tokenizer(input_text, return_token_type_ids=False, return_tensors="pt")
        raw["input_ids"] = tokenized_text["input_ids"][0]
        raw["attention_mask"] = tokenized_text["attention_mask"][0]
        raw["labels"] = int(raw["is_impossible"])

        raw["context"] = "".join(raw["context"].split("\n"))

        return raw

    with predict_args.main_process_first():
        raw_test_datasets = raw_test_datasets.map(preprocess, num_proc=4)
        raw_test_datasets = raw_test_datasets.filter(
            lambda x: len(x["input_ids"]) <= tokenizer.model_max_length,
            num_proc=4,
        )
        test_datasets = raw_test_datasets.shuffle()

    # [NOTE]: load metrics & set Trainer arguments
    accuracy = load("evaluate-metric/accuracy")

    def metrics(evaluation_result: EvalPrediction) -> Dict[str, float]:
        """_metrics_
            evaluation과정에서 모델의 성능을 측정하기 위한 metric을 수행하는 함수 입니다.
            이 함수는 Trainer에 의해 실행되며 Huggingface의 Evaluate 페키로 부터
            각종 metric을 전달받아 계산한 뒤 결과를 반환합니다.

        Args:
            evaluation_result (EvalPrediction): Trainer.evaluation_loop에서 model을 통해 계산된
            logits과 label을 전달받습니다.

        Returns:
            Dict[str, float]: metrics 계산결과를 dict로 반환합니다.
        """

        metrics_result = dict()

        predictions = evaluation_result.predictions
        references = evaluation_result.label_ids

        predictions = np.argmax(predictions, axis=-1)

        accuracy_result = accuracy._compute(predictions=predictions, references=references)

        metrics_result["accuracy"] = accuracy_result["accuracy"]
        return metrics_result

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    if predict_args.torch_compile:
        model = torch.compile(
            model,
            backend=predict_args.torch_compile_backend,
            mode=predict_args.torch_compile_mode,
        )
        torch._dynamo.config.verbose = True
        torch._dynamo.config.suppress_errors = True

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=test_datasets,
        args=predict_args,
        compute_metrics=metrics,
        data_collator=collator,
    )

    predict(trainer, test_datasets, tokenizer)


@torch.no_grad()
def predict(trainer: Trainer, test_data: Dataset) -> None:
    """_predict_
        Trainer를 전달받아 Trainer.predict을 실행시키는 함수입니다.
        이때 Trainer의 Predict이 실행되며 model.generator를 실행시키기 위해
        arg값의 predict_with_generater값을 강제로 True로 변환시킵니다.

        True로 변환시키면 model.generator에서 BeamSearch를 진행해 더 질이 좋은 결과물을 만들 수 있습니다.
    Args:
        trainer (Trainer): Huggingface의 torch Trainer를 전달받습니다.
        test_data (Dataset): 검증을 하기 위한 Data를 전달받습니다.
    """
    output = trainer.predict(test_data)

    softmax: np.ndarray = lambda x: np.exp(x) / np.sum(np.exp(x))  # NOTE: 1차원 softmax
    predictions = np.apply_along_axis(softmax, 1, output.predictions)
    # 그렇기 때문에 apply를 이용해 각 열에 적용시켜야 함.
    bool_pred = predictions.argmax(-1).astype(bool)

    test_data = test_data.remove_columns(["is_impossible", "input_ids", "attention_mask"])

    test_data = test_data.add_column(name="pred", column=bool_pred)
    test_data = test_data.add_column(name="impossible_prob", column=predictions[:, 0])
    test_data = test_data.add_column(name="possible_prob", column=predictions[:, 1])

    save_path = os.path.join(trainer.args.save_dir, "predict_result.json")
    test_data.to_json(save_path, force_ascii=False)

    save_path = os.path.join(trainer.args.save_dir, "predict_result.csv")
    df_data = test_data.to_pandas()
    df_data.to_csv(save_path)

    for metric_name, score in output.metrics.items():
        print(f"{metric_name:>15}: {score:.4f}")


if "__main__" in __name__:
    parser = HfArgumentParser([PredictArguments])
    main(parser)
