# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fine-tune FormLM to suggest block type."""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from datasets import load_metric, Dataset
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from models import FormLMConfig, FormLMForSequenceClassification, FormLMTokenizer, FormLinearize, BLOCK_TYPE_NAME_TO_IDX
from utils.helper import load_json

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    data_root_dir: Optional[str] = field(
        default="./Data", metadata={"help": "The root directory of the dataset."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="facebook/bart-large",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Select model with validation set
    training_args.load_best_at_end = True
    training_args.metric_for_best_model = "eval_macro_f1"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Prepare data from the raw datasets.
    input_column = "form"
    label_column = "label"
    form_linearizer = FormLinearize()
    raw_datasets = {}
    if training_args.do_train:
        train_dataset = {
            input_column: [],
            label_column: []
        }
        for _, _, files in os.walk(os.path.join(data_args.data_root_dir, "train")):
            for file in files:
                data = load_json(os.path.join(data_args.data_root_dir, "train", file))
                inputs, labels = form_linearizer.linearize_form_for_block_type_classification(data)
                train_dataset[input_column].extend(inputs)
                train_dataset[label_column].extend(labels)
        raw_datasets["train"] = Dataset.from_dict(train_dataset)
    if training_args.do_eval:
        validation_dataset = {
            input_column: [],
            label_column: []
        }
        for _, _, files in os.walk(os.path.join(data_args.data_root_dir, "dev")):
            for file in files:
                data = load_json(os.path.join(data_args.data_root_dir, "dev", file))
                inputs, labels = form_linearizer.linearize_form_for_block_type_classification(data)
                validation_dataset[input_column].extend(inputs)
                validation_dataset[label_column].extend(labels)
        raw_datasets["validation"] = Dataset.from_dict(validation_dataset)
    if training_args.do_predict:
        test_dataset = {
            input_column: [],
            label_column: []
        }
        for _, _, files in os.walk(os.path.join(data_args.data_root_dir, "test")):
            for file in files:
                data = load_json(os.path.join(data_args.data_root_dir, "test", file))
                inputs, labels = form_linearizer.linearize_form_for_block_type_classification(data)
                test_dataset[input_column].extend(inputs)
                test_dataset[label_column].extend(labels)
        raw_datasets["test"] = Dataset.from_dict(test_dataset)

    # Load pretrained model and tokenizer
    config = FormLMConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        add_structural_bias=True,
        add_distance_decay=True,
        num_labels=len(BLOCK_TYPE_NAME_TO_IDX) - 1,  # We exclude section block in this task.
        cache_dir=model_args.cache_dir,
    )
    tokenizer = FormLMTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.truncation_side = 'left'  # Keep the nearer context.
    model = FormLMForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples[input_column], padding=padding, max_length=max_seq_length, truncation=True)
        result["labels"] = examples[label_column]

        return result

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(desc="dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(desc="dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    acc_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions.argmax(1)
        labels = p.label_ids
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        micro_f1 = f1_metric.compute(predictions=preds, references=labels, average="micro")["f1"]

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1
        }

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint,
            ignore_keys_for_eval=["encoder_last_hidden_state"]
        )
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Testing
    if training_args.do_predict:
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict",
                                          ignore_keys=["encoder_last_hidden_state"])
        metrics = predict_results.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        trainer.log(metrics)


if __name__ == "__main__":
    main()
