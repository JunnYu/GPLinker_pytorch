import json
from argparse import Namespace as Config
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from models import get_auto_model
from utils.collate import DataCollatorForGPLinker, DataCollatorForTPLinkerPlus
from utils.postprocess import postprocess_gplinker, postprocess_tplinker_plus


class DummpyDataset(Dataset):
    def __init__(self, file, tokenizer, max_length=512, debug=False):
        super().__init__()
        self.data = self.load_data(file)
        if debug:
            self.data = self.data[:256]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        data = self.data[index]
        out = self.tokenizer(
            data["text"],
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            padding=False,
        )
        out["text"] = data["text"]
        return out

    def __len__(self):
        return len(self.data)

    def load_data(self, filename):
        """加载数据
        单条格式：{'text': text, 'spo_list': [(s, p, o)]}
        """
        D = []
        with open(filename, "r", encoding="utf8") as f:
            for l in f:
                l = json.loads(l)
                D.append(
                    {
                        "text": l["text"],
                        "spo_list": [
                            (spo["subject"], spo["predicate"], spo["object"])
                            for spo in l["spo_list"]
                        ],
                    }
                )
        return D


def fire(args):
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

    # model & tokenizer
    model_cls = get_auto_model(args.model_type, args.method)
    model = model_cls.from_pretrained(
        args.model_path, predicate2id=predicate2id)
    model.eval()
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = DummpyDataset(
        args.in_file, tokenizer, max_length=args.max_length, debug=args.debug
    )
    collate_cls = (
        DataCollatorForGPLinker
        if args.method == "gplinker"
        else DataCollatorForTPLinkerPlus
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_cls(tokenizer),
        num_workers=args.num_workers,
    )

    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Evaluating: ",
        ):
            offset_mappings = batch.pop("offset_mapping")
            texts = batch.pop("text")
            batch.to(args.device)
            outputs = model(**batch)[0]
            if args.method == "gplinker":
                outputs_gathered = postprocess_gplinker(
                    args, outputs, offset_mappings, texts, threshold=0
                )
            elif args.method == "tplinker_plus":
                outputs_gathered = postprocess_tplinker_plus(
                    args, outputs, offset_mappings, texts, batch["input_ids"].size(
                        1)
                )
            else:
                raise ValueError(
                    "args.method should be chosen from ['gplinker', 'tplinker_plus']!"
                )
            all_predictions.extend(outputs_gathered)

    X, Y, Z = 1e-10, 1e-10, 1e-10
    with open(args.out_file, "w", encoding="utf-8") as f:
        for preds, raw_data in zip(all_predictions, dataloader.dataset.data):
            R = set(preds)
            T = set(raw_data["spo_list"])
            X += len(R & T)
            Y += len(R)
            Z += len(T)

            s = json.dumps(
                {
                    "text": raw_data["text"],
                    "spo_list": list(T),
                    "spo_list_pred": list(R),
                    "new": list(R - T),
                    "lack": list(T - R),
                },
                ensure_ascii=False,
            )
            f.write(s + "\n")

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    retults = {"f1": f1, "precision": precision, "recall": recall}
    pprint(retults)
    return retults


if __name__ == "__main__":
    debug = True
    use_gpu = False
    args = Config(
        debug=debug,
        model_type="bert",
        method="gplinker",
        batch_size=32,
        max_length=128,
        num_workers=6,
        in_file="data/dev_data.json",
        out_file="preds.json" if not debug else "debug.json",
        model_path="outputs/bert-hfl_chinese-roberta-wwm-ext/ckpt/step-10804-spo-f1-0.81283402101807955",
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if use_gpu
        else "cpu",
    )
    fire(args)
