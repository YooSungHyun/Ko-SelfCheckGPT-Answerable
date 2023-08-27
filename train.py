# for type annotation
from argparse import Namespace
from typing import Any, Dict

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from evaluate import load
from setproctitle import setproctitle
from transformers import (
    DataCollatorWithPadding,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction
from utils import clean_unicode, trim_text

PASSAGE_QUESTION_TOKEN = "[unused0]"


def main(parser: HfArgumentParser) -> None:
    train_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle(train_args.run_name)
    set_seed(train_args.seed)

    model_name_or_path = "klue/roberta-large"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name_or_path)
    model = RobertaForSequenceClassification.from_pretrained(model_name_or_path)
    model.config.num_labels = 2

    klue_datasets = load_dataset("klue", "mrc")
    raw_train_datasets = load_dataset("json", data_files="./data/train/train_data.json", split="all")
    raw_valid_datasets = load_dataset("json", data_files="./data/validation/valid_data.json", split="all")

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

        return raw

    klue_datasets = klue_datasets.map(preprocess, remove_columns=klue_datasets["train"].column_names)
    klue_datasets = klue_datasets.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length)
    true_datasets = klue_datasets.filter(lambda x: x["labels"] == 1)
    false_datasets = klue_datasets.filter(lambda x: x["labels"] == 0)
    train_sampling_count = min(len(true_datasets["train"]), len(false_datasets["train"]))
    eval_sampling_count = min(len(true_datasets["validation"]), len(false_datasets["validation"]))
    train_true_datasets = Dataset.from_dict(true_datasets["train"].shuffle()[:train_sampling_count])
    train_false_datasets = Dataset.from_dict(false_datasets["train"].shuffle()[:train_sampling_count])
    eval_true_datasets = Dataset.from_dict(true_datasets["validation"].shuffle()[:eval_sampling_count])
    eval_false_datasets = Dataset.from_dict(false_datasets["validation"].shuffle()[:eval_sampling_count])
    raw_train_datasets = raw_train_datasets.map(preprocess, remove_columns=raw_train_datasets.column_names)
    raw_valid_datasets = raw_valid_datasets.map(preprocess, remove_columns=raw_valid_datasets.column_names)
    train_datasets = concatenate_datasets([train_true_datasets, train_false_datasets, raw_train_datasets]).shuffle()
    eval_datasets = concatenate_datasets([eval_true_datasets, eval_false_datasets, raw_valid_datasets]).shuffle()

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

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        args=train_args,
        compute_metrics=metrics,
        data_collator=collator,
    )

    # [NOTE]: run train, eval, predict
    if train_args.do_train:
        train(trainer, train_args)
    if train_args.do_eval:
        eval(trainer, eval_datasets)


def train(trainer: Trainer, args: Namespace) -> None:
    """_train_
        Trainer를 전달받아 Trainer.train을 실행시키는 함수입니다.
        학습이 끝난 이후 학습 결과 그리고 최종 모델을 저장하는 기능도 합니다.

        만약 학습을 특정 시점에 재시작 하고 싶다면 Seq2SeqTrainingArgument의
        resume_from_checkpoint을 True혹은 PathLike한 값을 넣어주세요.

        - huggingface.trainer.checkpoint
        https://huggingface.co/docs/transformers/main_classes/trainer#checkpoints

    Args:
        trainer (Trainer): Huggingface의 torch Trainer를 전달받습니다.
        args (Namespace): Seq2SeqTrainingArgument를 전달받습니다.
    """
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = outputs.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(args.output_dir)


def eval(trainer: Trainer, eval_data: Dataset) -> None:
    """_eval_
        Trainer를 전달받아 Trainer.eval을 실행시키는 함수입니다.
    Args:
        trainer (Trainer): Huggingface의 torch Trainer를 전달받습니다.
        eval_data (Dataset): 검증을 하기 위한 Data를 전달받습니다.
    """
    trainer.evaluate(eval_data)


if "__main__" in __name__:
    parser = HfArgumentParser([TrainingArguments])

    main(parser)
