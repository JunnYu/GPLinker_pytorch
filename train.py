import json
import logging
import math
import os
from pprint import pformat

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from fastcore.all import *
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler, set_seed

from models import get_auto_model
from utils.args import parse_args
from utils.data import get_dataloader_and_dataset
from utils.postprocess import postprocess_gplinker, postprocess_tplinker_plus
from utils.utils import get_writer, try_remove_old_ckpt, write_json

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    args,
    model,
    dev_dataloader,
    accelerator,
    global_steps=0,
    threshold=0,
    write_predictions=True,
):
    model.eval()
    all_predictions = []
    for batch in tqdm(
        dev_dataloader,
        disable=not accelerator.is_local_main_process,
        desc="Evaluating: ",
        leave=False,
    ):
        offset_mappings = batch.pop("offset_mapping")
        texts = batch.pop("text")
        outputs = model(**batch)[0]
        if args.method == "gplinker":
            outputs_gathered = postprocess_gplinker(
                args, accelerator.gather(
                    outputs), offset_mappings, texts, threshold
            )
        elif args.method == "tplinker_plus":
            outputs_gathered = postprocess_tplinker_plus(
                args,
                accelerator.gather(outputs),
                offset_mappings,
                texts,
                batch["input_ids"].size(1),
            )
        else:
            raise ValueError(
                "args.method should be chosen from ['gplinker', 'tplinker_plus']!"
            )
        all_predictions.extend(outputs_gathered)

    X, Y, Z = 1e-10, 1e-10, 1e-10
    if write_predictions:
        pred_dir = os.path.join(args.output_dir, "preds")
        os.makedirs(pred_dir, exist_ok=True)
        pred_file = os.path.join(
            pred_dir, f"{global_steps}_step_preds_{args.method}.json")
        f = open(pred_file, "w", encoding="utf-8")
    for preds, golds, text in zip(
        all_predictions,
        dev_dataloader.dataset.raw_data["spo_list"],
        dev_dataloader.dataset.raw_data["text"],
    ):
        R = set(preds)
        T = set([tuple(g) for g in golds])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        if write_predictions:
            s = json.dumps(
                {
                    "text": text,
                    "spo_list": list(T),
                    "spo_list_pred": list(R),
                    "new": list(R - T),
                    "lack": list(T - R),
                },
                ensure_ascii=False,
            )
            f.write(s + "\n")
    if write_predictions:
        f.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    model.train()

    return {"f1": f1, "precision": precision, "recall": recall}


def main():
    args = parse_args()
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "run.log"),
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    predicate2id = {}
    id2predicate = {}
    with open("data/all_50_schemas", "r", encoding="utf-8") as f:
        for l in f:
            l = json.loads(l)
            if l["predicate"] not in predicate2id:
                id2predicate[len(predicate2id)] = l["predicate"]
                predicate2id[l["predicate"]] = len(predicate2id)
    args.predicate2id = predicate2id
    args.id2predicate = id2predicate
    args.num_labels = len(id2predicate)

    if args.method == "tplinker_plus":
        link_types = [
            "SH2OH",  # subject head to object head
            "OH2SH",  # object head to subject head
            "ST2OT",  # subject tail to object tail
            "OT2ST",  # object tail to subject tail
        ]
        tags = []
        for lk in link_types:
            for rel in predicate2id.keys():
                tags.append("=".join([rel, lk]))
        tags.append("DEFAULT=EH2ET")
        args.tag2id = {t: idx for idx, t in enumerate(tags)}
        args.id2tag = {idx: t for t, idx in args.tag2id.items()}

    tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name
                                              )
    model = get_auto_model(args.model_type, args.method).from_pretrained(
        args.pretrained_model_name_or_path,
        predicate2id=predicate2id,
        cache_dir=args.model_cache_dir,
    )

    (train_dataloader, dev_dataloader) = get_dataloader_and_dataset(
        args,
        tokenizer,
        predicate2id=predicate2id,
        use_fp16=accelerator.use_fp16,
        text_column_name="text",
        label_column_name="spo_list",
    )

    no_decay = ["bias", "LayerNorm.weight", "norm"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    model, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
    args.num_warmup_steps = (
        math.ceil(args.max_train_steps * args.num_warmup_steps_or_radios)
        if isinstance(args.num_warmup_steps_or_radios, float)
        else args.num_warmup_steps_or_radios
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    args.total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("********** Running training **********")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args.total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps),
        leave=False,
        disable=not accelerator.is_local_main_process,
        desc="Training: ",
    )
    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    max_f1 = 0.0
    writer = get_writer(args)
    model.train()

    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")

    write_json(vars(args), os.path.join(args.output_dir, "args.json"))
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs[0]
            loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            accelerator.backward(loss)

            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                accelerator.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1

                if args.logging_steps > 0 and global_steps % args.logging_steps == 0:
                    writer.add_scalar(
                        "lr", lr_scheduler.get_last_lr()[-1], global_steps
                    )
                    writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_steps,
                    )
                    logger.info(
                        "global_steps {} - lr: {:.8f}  loss: {:.8f}".format(
                            global_steps,
                            lr_scheduler.get_last_lr()[-1],
                            (tr_loss - logging_loss) / args.logging_steps,
                        )
                    )
                    accelerator.print(
                        "global_steps {} - lr: {:.8f}  loss: {:.8f}".format(
                            global_steps,
                            lr_scheduler.get_last_lr()[-1],
                            (tr_loss - logging_loss) / args.logging_steps,
                        )
                    )
                    logging_loss = tr_loss

                if (
                    args.save_steps > 0 and global_steps % args.save_steps == 0
                ) or global_steps == args.max_train_steps:
                    logger.info(
                        f"********** Evaluate Step {global_steps} **********")
                    accelerator.print("##--------------------- Dev")
                    logger.info("##--------------------- Dev")
                    dev_metric = evaluate(
                        args, model, dev_dataloader, accelerator, global_steps, 0, True
                    )
                    accelerator.print("-" * 80)
                    logger.info("-" * 80)
                    for k, v in dev_metric.items():
                        accelerator.print(f"{k} = {v}")
                        logger.info(f"{k} = {v}")
                        writer.add_scalar(
                            f"dev/{k}",
                            v,
                            global_steps,
                        )
                    accelerator.print("-" * 80)
                    logger.info("-" * 80)
                    accelerator.print("**--------------------- Dev End")
                    logger.info("**--------------------- Dev End")

                    f1 = dev_metric["f1"]
                    if f1 >= max_f1:
                        max_f1 = f1
                        savefile = Path(args.output_dir) / "val_results.txt"
                        savefile.write_text(
                            pformat(dev_metric), encoding="utf-8")

                    output_dir = os.path.join(
                        args.output_dir, "ckpt", f"step-{global_steps}-spo-f1-{f1}"
                    )

                    os.makedirs(output_dir, exist_ok=True)
                    accelerator.wait_for_everyone()
                    tokenizer.save_pretrained(output_dir)
                    accelerator.unwrap_model(model).save_pretrained(
                        output_dir, save_function=accelerator.save
                    )
                    try_remove_old_ckpt(args.output_dir, topk=args.topk)
                    logger.info("*************************************")

            if global_steps >= args.max_train_steps:
                return


if __name__ == "__main__":
    main()
