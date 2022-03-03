import json
import logging
import math
import os
from pprint import pformat

import datasets
import torch
import transformers
from accelerate import Accelerator
from fastcore.all import *
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler, set_seed

from models import get_auto_model
from utils.args import parse_args
from utils.data import get_dataloader_and_dataset
from utils.postprocess import DedupList, isin, postprocess_gplinker
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
        outputs_gathered = postprocess_gplinker(
            args,
            accelerator.gather(outputs),
            offset_mappings,
            texts,
            trigger=False,
            threshold=threshold,
        )
        all_predictions.extend(outputs_gathered)

    ex, ey, ez = 1e-10, 1e-10, 1e-10  # 事件级别
    ax, ay, az = 1e-10, 1e-10, 1e-10  # 论元级别

    if write_predictions:
        pred_dir = os.path.join(args.output_dir, "preds")
        os.makedirs(pred_dir, exist_ok=True)
        pred_file = os.path.join(pred_dir, f"{global_steps}_step_preds.json")
        f = open(pred_file, "w", encoding="utf-8")
    for pred_events, events, texts in zip(
        all_predictions,
        dev_dataloader.dataset.raw_data["events"],
        dev_dataloader.dataset.raw_data["text"],
    ):
        R, T = DedupList(), DedupList()
        # 事件级别
        for event in pred_events:
            if any([argu[1] == "触发词" for argu in event]):
                R.append(list(sorted(event)))
        for event in events:
            T.append(list(sorted(event)))
        for event in R:
            if event in T:
                ex += 1
        ey += len(R)
        ez += len(T)
        # 论元级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            for argu in event:
                if argu[1] != "触发词":
                    R.append(argu)
        for event in events:
            for argu in event:
                if argu[1] != "触发词":
                    T.append(argu)
        for argu in R:
            if argu in T:
                ax += 1
        ay += len(R)
        az += len(T)

        if write_predictions:
            event_list = DedupList()
            for event in pred_events:
                final_event = {
                    "event_type": event[0][0], "arguments": DedupList()}
                for argu in event:
                    if argu[1] != "触发词":
                        final_event["arguments"].append(
                            {"role": argu[1], "argument": argu[2]}
                        )
                event_list = [
                    event for event in event_list if not isin(event, final_event)
                ]
                if not any([isin(final_event, event) for event in event_list]):
                    event_list.append(final_event)

            l = json.dumps(
                {"text": texts, "event_list": event_list}, ensure_ascii=False
            )
            f.write(l + "\n")

    e_f1, e_pr, e_rc = 2 * ex / (ey + ez), ex / ey, ex / ez
    a_f1, a_pr, a_rc = 2 * ax / (ay + az), ax / ay, ax / az

    if write_predictions:
        f.close()

    model.train()

    return {
        "e_f1": e_f1,
        "e_pr": e_pr,
        "e_rc": e_rc,
        "a_f1": a_f1,
        "a_pr": a_pr,
        "a_rc": a_rc,
    }


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

    labels = []
    with open(os.path.join(args.file_path, "duee_event_schema.json"), "r", encoding="utf-8") as f:
        for l in f:
            l = json.loads(l)
            t = l["event_type"]
            for r in ["触发词"] + [s["role"] for s in l["role_list"]]:
                labels.append((t, r))
    args.labels = labels
    args.num_labels = len(labels)

    tokenizer_name = (
        args.tokenizer_name
        if args.tokenizer_name is not None
        else args.pretrained_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = get_auto_model(args.model_type).from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=args.num_labels,
        cache_dir=args.model_cache_dir,
        use_efficient=args.use_efficient,
    )

    (train_dataloader, dev_dataloader) = get_dataloader_and_dataset(
        args,
        tokenizer,
        labels,
        use_fp16=accelerator.use_fp16,
        text_column_name="text",
        label_column_name="events",
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
    max_e_f1 = 0.0
    max_a_f1 = 0.0
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
                    args.save_steps > 0 and global_steps % args.save_steps == 0 and loss <= 3.  # 当loss太大时，预测是没有意义的。
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

                    e_f1, a_f1 = dev_metric["e_f1"], dev_metric["a_f1"]

                    if e_f1 >= max_e_f1:
                        max_e_f1 = e_f1
                        savefile = Path(args.output_dir) / "e_val_results.txt"
                        savefile.write_text(
                            pformat(dev_metric), encoding="utf-8")

                        e_output_dir = os.path.join(
                            args.output_dir,
                            "e",
                            "ckpt",
                            f"step-{global_steps}-f1-{e_f1}",
                        )
                        os.makedirs(e_output_dir, exist_ok=True)
                        accelerator.wait_for_everyone()
                        tokenizer.save_pretrained(e_output_dir)
                        accelerator.unwrap_model(model).save_pretrained(
                            e_output_dir, save_function=accelerator.save
                        )
                        try_remove_old_ckpt(
                            os.path.join(args.output_dir, "e"), topk=args.topk
                        )

                    if a_f1 >= max_a_f1:
                        max_a_f1 = a_f1
                        savefile = Path(args.output_dir) / "a_val_results.txt"
                        savefile.write_text(
                            pformat(dev_metric), encoding="utf-8")

                        a_output_dir = os.path.join(
                            args.output_dir,
                            "a",
                            "ckpt",
                            f"step-{global_steps}-f1-{a_f1}",
                        )
                        os.makedirs(a_output_dir, exist_ok=True)
                        accelerator.wait_for_everyone()
                        tokenizer.save_pretrained(a_output_dir)
                        accelerator.unwrap_model(model).save_pretrained(
                            a_output_dir, save_function=accelerator.save
                        )
                        try_remove_old_ckpt(
                            os.path.join(args.output_dir, "a"), topk=args.topk
                        )

                    logger.info("*************************************")

            if global_steps >= args.max_train_steps:
                return


if __name__ == "__main__":
    main()
