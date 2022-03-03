import logging
import random
import os

from datasets import Dataset
from torch.utils.data import DataLoader

from utils.collate import DataCollatorForGPLinkerDuEE

logger = logging.getLogger(__name__)


def duee_v1_process(example):
    events = []
    for e in example["event_list"]:
        offset1 = len(e["trigger"]) - len(e["trigger"].lstrip())
        events.append(
            [
                [
                    e["event_type"],
                    "触发词",
                    e["trigger"],
                    str(e["trigger_start_index"] + offset1)
                    + ";"
                    + str(e["trigger_start_index"] + offset1 + len(e["trigger"].strip())),
                ]
            ]
        )
        for a in e["arguments"]:
            offset2 = len(a["argument"]) - len(a["argument"].lstrip())
            events[-1].append(
                [
                    e["event_type"],
                    a["role"],
                    a["argument"],
                    str(a["argument_start_index"] + offset2)
                    + ";"
                    + str(a["argument_start_index"] + offset2 + len(a["argument"].strip())),
                ]
            )
    del example["event_list"]
    return {"events": events}


def get_dataloader_and_dataset(
    args,
    tokenizer,
    labels2id,
    use_fp16=False,
    text_column_name="text",
    label_column_name="events",
):
    train_raw_dataset = Dataset.from_json(os.path.join(args.file_path, "duee_train.json"))
    train_ds = train_raw_dataset.map(duee_v1_process)
    dev_raw_dataset = Dataset.from_json(os.path.join(args.file_path, "duee_dev.json"))
    dev_ds = dev_raw_dataset.map(duee_v1_process)

    def tokenize_and_align_train_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
        )
        labels = []
        for b, events in enumerate(examples[label_column_name]):
            argu_labels = {}
            head_labels = []
            tail_labels = []
            for event in events:
                for i1, (event_type1, rol1, word1, span1) in enumerate(event):
                    tp1 = labels2id.index((event_type1, rol1))
                    head1, tail1 = list(map(int, span1.split(";")))
                    tail1 = tail1 - 1
                    try:
                        h1 = tokenized_inputs.char_to_token(b, head1)
                        t1 = tokenized_inputs.char_to_token(b, tail1)
                    except Exception as e:
                        logger.info(f"{e} char_to_token error!")
                        continue
                    if h1 is None or t1 is None:
                        logger.info("find None!")
                        continue
                    if tp1 not in argu_labels:
                        argu_labels[tp1] = [tp1]
                    argu_labels[tp1].extend([h1, t1])

                    for i2, (event_type2, rol2, word2, span2) in enumerate(event):
                        if i2 > i1:
                            head2, tail2 = list(map(int, span2.split(";")))
                            tail2 = tail2 - 1
                            try:
                                h2 = tokenized_inputs.char_to_token(b, head2)
                                t2 = tokenized_inputs.char_to_token(b, tail2)
                            except Exception as e:
                                logger.info("char_to_token error!")
                                continue
                            if h2 is None or t2 is None:
                                logger.info("find None!")
                                continue
                            hl = [min(h1, h2), max(h1, h2)]
                            tl = [min(t1, t2), max(t1, t2)]
                            if hl not in head_labels:
                                head_labels.append(hl)
                            if tl not in tail_labels:
                                tail_labels.append(tl)

            argu_labels = list(argu_labels.values())
            labels.append(
                {
                    "argu_labels": argu_labels if len(argu_labels)>0 else [[0,0,0]],
                    "head_labels": head_labels if len(head_labels)>0 else [[0,0]],
                    "tail_labels": tail_labels if len(tail_labels)>0 else [[0,0]]
                }
            )

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=False,
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        return tokenized_inputs

    train_dataset = train_ds.map(
        tokenize_and_align_train_labels,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Running tokenizer on train dataset",
        new_fingerprint=f"duee-train-{args.model_type}-{args.max_length}-{args.model_weights}",
    )
    dev_dataset = dev_ds.map(
        tokenize,
        batched=True,
        remove_columns=["id", "events"],  # 保留text
        desc="Running tokenizer on dev dataset",
        new_fingerprint=f"duee-dev-{args.model_type}-{args.max_length}-{args.model_weights}",
    )
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set:")
        for k, v in train_dataset[index].items():
            logger.info(f"{k} = {v}")

    data_collator = DataCollatorForGPLinkerDuEE(
        tokenizer,
        pad_to_multiple_of=(8 if use_fp16 else None),
        num_labels=args.num_labels,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
    )
    dev_dataset.raw_data = dev_ds

    return train_dataloader, dev_dataloader
