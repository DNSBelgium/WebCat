import argparse
import copy
import datetime
import math
import random
from dataclasses import dataclass
from typing import Literal, overload
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn.metrics as metrics
import torch
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import DEVICE
from preprocess import PreprocessedTrainingData, PreprocessedInputs
from utils import move_to
from xlmr import CustomXlmRoberta, create_bert


class TrainingValidator:
    """
    Class that calculates the validation score during the training process and keeps track of the best state dict.
    """
    validation_score_opt: float
    binary_threshold_opt: float | None
    validation_set: PreprocessedInputs
    label_amount: int
    best_state_dict: dict

    def __init__(self, validation_set: PreprocessedInputs, label_amount: int):
        self.validation_set = validation_set
        self.label_amount = label_amount
        self.validation_score_opt = -1
        self.binary_threshold_opt = None

    def step(self, model: CustomXlmRoberta) -> float:
        """
        Calculates validation score of a model and replaces best_state_dict if it performs better than the previously
        highest validation score.
        """
        model.eval()

        # noinspection PyTypeChecker
        output_type: Literal["binary", "multi"] = "binary" if self.label_amount == 2 else "multi"
        pred_df = make_predictions(self.validation_set, model, None, output_type, False, None, None, False)
        if self.label_amount == 2:
            preds = pred_df["prediction"].to_numpy()
        else:
            preds = pred_df["predicted_label"].to_numpy()

        real_labels = pred_df["true_label"].to_numpy()

        if self.label_amount == 2:
            precision, recall, thresholds = precision_recall_curve(real_labels, preds)
            score = metrics.auc(recall, precision)
        else:
            score = f1_score(real_labels, preds, average="weighted")
            precision, recall, thresholds = (None, None, None)

        if score > self.validation_score_opt:
            self.validation_score_opt = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            if precision is not None and recall is not None and thresholds is not None:
                f1s = 2 * precision * recall / (precision + recall)
                argmax = np.nanargmax(f1s)
                self.binary_threshold_opt = thresholds[argmax]

        model.train()

        return score


@dataclass
class ValidationResult:
    epoch: int
    batch: int
    f1: float


def create_optimiser_and_scheduler(
        model: CustomXlmRoberta,
        dataloader_train: DataLoader,
        learning_rate: float,
        epochs: int
) -> Tuple[Optimizer, LambdaLR]:
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    return optimizer, scheduler


def run_epoch(
        training_dl: DataLoader,
        model: CustomXlmRoberta,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        epoch: int,
        validator: TrainingValidator) -> list[ValidationResult]:
    """
    Runs a training epoch.

    :param training_dl: DataLoader for training data
    :param model: XLM-R model
    :param optimizer: Optimizer (from create_optimizer_and_scheduler)
    :param scheduler: Scheduler (from create_optimizer_and_scheduler)
    :param epoch: Number of the current epoch
    :param validator: TrainingValidator instance that keeps track of the best configuration during this training run
    :return: Progress of the validation score during the epoch
    """
    model.train()
    loss_train_total = 0

    progress_bar = tqdm(training_dl, desc="Epoch {:1d}".format(epoch), leave=False, disable=False)

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    batch_index = -1
    dl_len = len(training_dl)
    one_tenth = math.ceil(dl_len / 10)

    val_results = []

    f1 = None

    for batch in progress_bar:
        batch_index += 1

        model.zero_grad()
        inputs = move_to(batch, DEVICE)

        del inputs["visit_ids"]

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()

        if scaler is not None:
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        if batch_index > 0 and (batch_index % one_tenth == 0 or batch_index == dl_len - 1) \
                and (epoch > 2 or batch_index == dl_len - 1):
            del batch
            del inputs
            torch.cuda.empty_cache()
            f1 = validator.step(model)
            progress_bar.set_postfix({"validation score": "{:.3f}".format(f1)})
            val_results.append(ValidationResult(epoch=epoch, batch=batch_index, f1=f1))

    loss_train_avg = loss_train_total / len(training_dl)
    tqdm.write(f"Training loss: {loss_train_avg}")
    tqdm.write(f"Validation score (end of epoch): {f1}")

    return val_results


def make_dataloader_of_preprocessed(inputs: PreprocessedInputs,
                                    batch_size: int,
                                    random_sampling: bool = True) -> DataLoader:
    return DataLoader(inputs, sampler=None if not random_sampling else RandomSampler(inputs), batch_size=batch_size)


def seed_random(seed_val: int):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def train_bert(inputs: PreprocessedTrainingData,
               learning_rate: float,
               epochs: int,
               batch_size: int) -> tuple[ColumnTransformer, dict, float | None, list[ValidationResult]]:
    """
    Train a new model.

    :return: (fitted ColumnTransformer, model state dict, binary threshold for binary models, progress of the validation
              score during the training)
    """
    print("Starting training process.")
    col_transf = inputs.col_transformer()
    data_train = make_dataloader_of_preprocessed(inputs.training, batch_size)
    label_amount = len(np.unique(inputs.training.y()))
    bert = create_bert(label_amount)
    bert.train()
    optimizer, scheduler = create_optimiser_and_scheduler(bert, data_train, learning_rate, epochs)

    early_stopper = TrainingValidator(inputs.validation, label_amount)
    val_results = []

    for epoch in tqdm(range(1, epochs + 1)):
        print("running epoch", epoch, "at", datetime.datetime.now())
        val_results.extend(run_epoch(data_train, bert, optimizer, scheduler, epoch, early_stopper))
    print("Done.")
    return col_transf, early_stopper.best_state_dict, early_stopper.binary_threshold_opt, val_results


def load_model(path: str, map_location=None) -> tuple[ColumnTransformer, CustomXlmRoberta, float | None, LabelEncoder]:
    col_transf, state_dict, binary_threshold, label_encoder = torch.load(path, map_location)
    bert = create_bert(len(label_encoder.classes_), False)
    bert.load_state_dict(state_dict)
    bert = bert.to(DEVICE)
    return col_transf, bert, binary_threshold, label_encoder


@overload
def make_predictions(inputs: PreprocessedInputs, model: CustomXlmRoberta, label_encoder: LabelEncoder | None,
                     output_type: Literal["multi", "binary"],
                     include_entropy: bool,
                     binary_threshold: float | None, output_path: str,
                     progress_bar: bool) -> None:
    ...


@overload
def make_predictions(inputs: PreprocessedInputs, model: CustomXlmRoberta, label_encoder: LabelEncoder | None,
                     output_type: Literal["multi", "binary"],
                     include_entropy: bool,
                     binary_threshold: float | None, output_path: None,
                     progress_bar: bool) -> pd.DataFrame:
    ...


def make_predictions(inputs: PreprocessedInputs, model: CustomXlmRoberta, label_encoder: LabelEncoder | None,
                     output_type: Literal["multi", "binary"],
                     include_entropy: bool,
                     binary_threshold: float | None, output_path: str | None,
                     progress_bar: bool) -> None | pd.DataFrame:
    """
    Make predictions using a model.

    :param inputs: Model inputs
    :param model: The model
    :param label_encoder: Optional LabelEncoder so the outputs can contain the textual labels
    :param output_type: "binary" for binary models, "multi" for multiclass models
    :param include_entropy: Should the prediction entropy be included in the output
    :param binary_threshold: Optionally, for binary models, the decision threshold
    :param output_path: Path to store predictions in (as Parquet file) - if None, predictions are returned as DataFrame
    :param progress_bar: Should a progress bar be displayed during inference time
    """
    pqwriter = None
    out_df = None

    model.eval()

    dl = make_dataloader_of_preprocessed(inputs, 200, False)
    with torch.no_grad():
        recombine_buffer = None

        for i, batch in (enumerate(tqdm(dl)) if progress_bar else enumerate(dl)):
            # Important: a website may be split into segments that are across different batches!
            # This must be kept in mind when modifying code that combines segment predictions.

            batch_visit_ids = batch["visit_ids"]
            del batch["visit_ids"]

            if "labels" in batch:
                batch_labels = batch["labels"].numpy()
                del batch["labels"]
            else:
                batch_labels = None

            batch = move_to(batch, DEVICE)

            outputs = model(**batch)
            logits = outputs["logits"].detach().cpu().numpy()
            probabilities = scipy.special.softmax(logits, axis=1)

            # Assumption: websites are split in at most two segments. This is correct with the current
            # window function, but the following code needs to be modified if that changes.

            if recombine_buffer is None:
                to_recombine = probabilities
                batch_visit_ids_tr = batch_visit_ids
                batch_labels_tr = batch_labels
            else:
                b_probs, b_visits, b_labels = recombine_buffer
                to_recombine = np.vstack((b_probs, probabilities))
                batch_visit_ids_tr = np.concatenate((b_visits, batch_visit_ids))
                if batch_labels is not None:
                    batch_labels_tr = np.concatenate((b_labels, batch_labels))
                else:
                    batch_labels_tr = None

            if len(batch_visit_ids) > 1 and batch_visit_ids[-1] != batch_visit_ids[-2] and i < len(dl) - 1:
                # The last prediction output is the only segment of a website in this batch.
                # We don't yet know if the next batch has the second segment. For safety,
                # set the last prediction aside and combine it with the next batch output.
                # This does not need to be done for the last batch, of course.
                recombine_buffer = ([probabilities[-1]], [batch_visit_ids[-1]],
                                    None if batch_labels is None else [batch_labels[-1]])
                to_recombine = to_recombine[:-1]
                batch_visit_ids_tr = batch_visit_ids_tr[:-1]
                if batch_labels_tr is not None:
                    batch_labels_tr = batch_labels_tr[:-1]
            else:
                recombine_buffer = None

            visit_ids, recombined_probs, real_labels = recombine_segment_predictions(to_recombine,
                                                                                     batch_visit_ids_tr,
                                                                                     batch_labels_tr)

            if output_type == "multi":
                results = recombined_probs.argmax(axis=1)
                if label_encoder is not None:
                    labels = label_encoder.inverse_transform(results)
                else:
                    labels = results
                df_dict = {
                    "visit_id": visit_ids,
                    "predicted_label": labels
                }
            elif output_type == "binary":
                predictions = recombined_probs[:, 1]
                df_dict = {
                    "visit_id": visit_ids,
                    "prediction": predictions
                }
                if binary_threshold is not None:
                    decisions = predictions > binary_threshold
                    df_dict["decision"] = decisions
            else:
                raise ValueError(output_type)

            if include_entropy:
                entropies = entropy(recombined_probs, axis=1)
                df_dict["entropy"] = entropies

            if real_labels is not None:
                if label_encoder:
                    real_labels = label_encoder.inverse_transform(real_labels)
                df_dict["true_label"] = real_labels

            df = pd.DataFrame(df_dict)

            if output_path is not None:
                # noinspection PyArgumentList
                table = pa.Table.from_pandas(df)

                if pqwriter is None:
                    pqwriter = pq.ParquetWriter(output_path, table.schema)

                pqwriter.write_table(table)
            else:
                if out_df is None:
                    out_df = df
                else:
                    out_df = pd.concat((out_df, df), ignore_index=True)

    if pqwriter:
        pqwriter.close()

    if output_path is None:
        return out_df
    else:
        return None


@overload
def recombine_segment_predictions(preds: npt.NDArray[np.float32], groups: npt.NDArray, true_y: npt.NDArray) -> \
        tuple[list[str], npt.NDArray[np.float32], list[int]]:
    ...


@overload
def recombine_segment_predictions(preds: npt.NDArray[np.float32], groups: npt.NDArray, true_y: None) -> \
        tuple[list[str], npt.NDArray[np.float32], None]:
    ...


def recombine_segment_predictions(preds: npt.NDArray[np.float32], groups: npt.NDArray, true_y: npt.NDArray | None) \
        -> tuple[list[str], npt.NDArray[np.float32], list[int] | None]:
    """
    'Recombines' predictions: both the start and the end of the website text are used as model inputs, and a prediction
    is made on those texts separately. But we only want one prediction per website in the end, so those predictions
    have to be combined (by averaging).
    """
    assert len(preds) == len(groups)

    agg_vector = []
    predictions = []
    visit_ids = []
    if true_y is None:
        true_labels: list[int] | None = None
    else:
        true_labels = []
    # true_labels is generally None during predictions, the exception is during training, when predictions on the
    # validation set are generated for model selection

    for i, vector in enumerate(preds):
        if i == 0:
            agg_vector = [vector]
            continue

        if groups[i] == groups[i - 1]:
            agg_vector.append(vector)
        else:
            predictions.append(np.mean(np.array(agg_vector), axis=0))
            visit_ids.append(groups[i - 1])
            if true_labels is not None and true_y is not None:
                true_labels.append(true_y[i - 1])
            agg_vector = [vector]

    predictions.append(np.mean(np.array(agg_vector), axis=0))
    visit_ids.append(groups[-1])
    if true_labels is not None and true_y is not None:
        true_labels.append(true_y[-1])

    return visit_ids, np.vstack(predictions), true_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="command")
    sp.required = True
    sp_train = sp.add_parser("train")
    sp_predict = sp.add_parser("predict")
    sp_info = sp.add_parser("info")

    sp_train.add_argument("inputs")
    sp_train.add_argument("out")
    sp_train.add_argument("--batch-size", type=int, default=24)
    sp_train.add_argument("--epochs", type=int, default=4)
    sp_train.add_argument("--learning-rate", type=float, default=2e-5)
    sp_train.add_argument("--seed", default="random")

    sp_predict.add_argument("data")
    sp_predict.add_argument("model")
    sp_predict.add_argument("out")
    sp_predict.add_argument("--entropies", action="store_true")

    sp_info.add_argument("model")

    args = parser.parse_args()

    inputs: PreprocessedTrainingData | PreprocessedInputs

    if args.command == "train":
        if args.seed != "random":
            if args.seed.isnumeric():
                seed_random(int(args.seed))
            else:
                print("Seed must be 'random' or an integer.")
                return
        inputs = PreprocessedTrainingData.load(args.inputs)
        col_transformer, state_dict, binary_threshold, _ = train_bert(inputs, args.learning_rate, args.epochs,
                                                                      args.batch_size)
        torch.save((col_transformer, state_dict, binary_threshold, inputs.label_encoder()), args.out)
    elif args.command == "predict":
        _, model, binary_threshold, label_encoder = load_model(args.model)
        inputs = PreprocessedInputs.load(args.data)

        # noinspection PyTypeChecker
        output_type: Literal["binary", "multi"] = "binary" if len(label_encoder.classes_) == 2 else "multi"
        print("Starting predictions.")
        make_predictions(inputs, model, label_encoder, output_type, args.entropies, binary_threshold, args.out, True)
    elif args.command == "info":
        col_transf, state_dict, binary_threshold, label_encoder = torch.load(args.model)
        label_count = len(label_encoder.classes_)
        if label_count == 2:
            print("Binary model")
        else:
            print(f"Multiclass model ({label_count} labels)")

        if binary_threshold is not None:
            print(f"Positive classification threshold: {binary_threshold}")


if __name__ == "__main__":
    main()
