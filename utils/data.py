import logging
import random

from datasets import load_dataset
from torch.utils.data import DataLoader

from utils.collate import DataCollatorForGPLinker, DataCollatorForTPLinkerPlus

logger = logging.getLogger(__name__)


def process_train(ds, predicate2id):
    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i: i + n] == pattern:
                return i
        return -1

    def judge(example):
        spo_list = []
        for spo in example["spo_list"]:
            sub = search(spo["subject"], example["text"])
            obj = search(spo["object"], example["text"])
            if sub == -1 or obj == -1:
                continue
            else:
                spo_list.append([1])
        return len(spo_list) > 0

    def convert(example):
        spo_list = []
        for spo in example["spo_list"]:
            sub = search(spo["subject"], example["text"])
            pre = predicate2id[spo["predicate"]]
            obj = search(spo["object"], example["text"])
            if sub == -1 or obj == -1:
                continue
            else:
                spo_list.append(
                    [
                        sub,
                        sub + len(spo["subject"]) - 1,
                        pre,
                        obj,
                        obj + len(spo["object"]) - 1,
                    ]
                )

        assert len(spo_list) > 0
        return {"text": example["text"], "spo_list": spo_list}

    return ds.filter(judge).map(convert)


def process_dev(example):
    triplet = []
    for spo in example["spo_list"]:
        triplet.append(
            [
                spo["subject"],
                spo["predicate"],
                spo["object"],
            ]
        )

    return {"spo_list": triplet}


def get_dataloader_and_dataset(
    args,
    tokenizer,
    predicate2id,
    use_fp16=False,
    text_column_name="text",
    label_column_name="spo_list",
):

    ds = load_dataset("./data/spo.py", cache_dir=args.cache_dir)
    trains_ds = process_train(ds["train"], predicate2id=predicate2id)
    devs_ds = ds["validation"].map(process_dev)

    def tokenize_and_align_train_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
        )
        labels = []
        for i, spo_list in enumerate(examples[label_column_name]):
            spo = []
            for _sh, _st, p, _oh, _ot in spo_list:
                try:
                    sh = tokenized_inputs.char_to_token(i, _sh)
                    oh = tokenized_inputs.char_to_token(i, _oh)
                    st = tokenized_inputs.char_to_token(i, _st)
                    ot = tokenized_inputs.char_to_token(i, _ot)
                except:
                    logger.info("char_to_token error!")
                    continue
                if sh is None or oh is None or st is None or ot is None:
                    continue
                spo.append([sh, st, p, oh, ot])
            labels.append(spo)
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

    train_dataset = trains_ds.map(
        tokenize_and_align_train_labels,
        batched=True,
        remove_columns=trains_ds.column_names,
        desc="Running tokenizer on train dataset",
        new_fingerprint=f"train-{args.model_type}-{args.max_length}-{args.model_weights}",
    )
    dev_dataset = devs_ds.map(
        tokenize,
        batched=True,
        remove_columns=["spo_list", "id"],
        desc="Running tokenizer on dev dataset",
        new_fingerprint=f"dev-{args.model_type}-{args.max_length}-{args.model_weights}",
    )
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set:")
        for k, v in train_dataset[index].items():
            logger.info(f"{k} = {v}")

    if args.method == "gplinker":
        collate_cls = DataCollatorForGPLinker
    else:
        collate_cls = DataCollatorForTPLinkerPlus
    data_collator = collate_cls(
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
    dev_dataset.raw_data = devs_ds

    return train_dataloader, dev_dataloader
