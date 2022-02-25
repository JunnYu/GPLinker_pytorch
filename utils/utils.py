import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def write_json(data, file_path):
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_json(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def write_jsonl(data, file):
    with open(file, "w", encoding="utf8") as f:
        for each in data:
            f.write(json.dumps(each, ensure_ascii=False))
            f.write("\n")


def load_jsonl(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data


def load_jsonl_smart(file):
    with open(file, "r", encoding="utf8") as f:
        s = f.read()
    _dec = json.JSONDecoder()
    all_data = []
    while s.find("{") >= 0:
        s = s[s.find("{") :]
        obj, pos = _dec.raw_decode(s)
        if not pos:
            raise ValueError(f"no JSON object found at {pos}")
        all_data.append(obj)
        s = s[pos:]
    return all_data


def try_remove_old_ckpt(output_dir, topk=5):
    if topk <= 0:
        return
    p = Path(output_dir) / "ckpt"
    ckpts = sorted(
        p.glob("step-*"), key=lambda x: float(x.name.split("-")[-1]), reverse=True
    )
    if len(ckpts) > topk:
        shutil.rmtree(ckpts[-1])
        logger.info(f"remove old ckpt: {ckpts[-1]}")


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.log_dir)
    elif args.writer_type == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer
