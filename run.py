import json
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler
from models import GPLinker
from utils import DataGenerator, globalpointer_loss, sequence_padding, Tokenizer

def to_device(batch, device="cpu"):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = [vv.to(device) for vv in v]
    return out
    
def load_data(filename):
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
        
def train(model, device, train_generator, lr_scheduler, optimizer):
    model.train()
    progress_bar = tqdm(
        range(len(train_generator)),
        leave=False,
        desc="Training: ",
    )
    for batch in train_generator.forfit():
        batch = to_device(batch, device)
        labels = batch.pop("labels")
        output = model(**batch)
        loss = sum([globalpointer_loss(o, l) for o, l in zip(output, labels)])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if progress_bar.n >= len(train_generator):
            break

def main():
    efficient = False # 是否使用EfficientGlobalpointer
    epochs = 20
    maxlen = 128
    batch_size = 16 # 3090可设置的大一点。
    weight_decay = 0.01
    lr = 3e-5
    dict_path = "./chinese-roberta-wwm-ext/vocab.txt" # 预训练模型vocab.txt路径
    model_name_or_path = "hfl/chinese-roberta-wwm-ext" # 预训练模型权重路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_data = load_data("data/train_data.json")
    valid_data = load_data("data/dev_data.json")
    predicate2id, id2predicate = {}, {}

    with open("data/all_50_schemas", "r", encoding="utf8") as f:
        for l in f:
            l = json.loads(l)
            if l["predicate"] not in predicate2id:
                id2predicate[len(predicate2id)] = l["predicate"]
                predicate2id[l["predicate"]] = len(predicate2id)

    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i : i + n] == pattern:
                return i
        return -1

    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    model = GPLinker.from_pretrained(
        model_name_or_path, predicate2id=predicate2id, efficient=efficient
    )
    model.to(device)

    class data_generator(DataGenerator):
        """数据生成器"""

        def __iter__(self, random=False):
            batch_token_ids, batch_segment_ids = [], []
            batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
            for is_end, d in self.sample(random):
                token_ids, segment_ids = tokenizer.encode(d["text"], maxlen=maxlen)
                # 整理三元组 {(s, o, p)}
                spoes = set()
                for s, p, o in d["spo_list"]:
                    s = tokenizer.encode(s)[0][1:-1]
                    p = predicate2id[p]
                    o = tokenizer.encode(o)[0][1:-1]
                    sh = search(s, token_ids)
                    oh = search(o, token_ids)
                    if sh != -1 and oh != -1:
                        spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))
                # 构建标签
                entity_labels = [set() for _ in range(2)]
                head_labels = [set() for _ in range(len(predicate2id))]
                tail_labels = [set() for _ in range(len(predicate2id))]
                for sh, st, p, oh, ot in spoes:
                    entity_labels[0].add((sh, st))
                    entity_labels[1].add((oh, ot))
                    head_labels[p].add((sh, oh))
                    tail_labels[p].add((st, ot))
                for label in entity_labels + head_labels + tail_labels:
                    if not label:  # 至少要有一个标签
                        label.add((0, 0))  # 如果没有则用0填充
                entity_labels = sequence_padding([list(l) for l in entity_labels])
                head_labels = sequence_padding([list(l) for l in head_labels])
                tail_labels = sequence_padding([list(l) for l in tail_labels])
                # 构建batch
                L = len(token_ids)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_entity_labels.append(entity_labels)
                batch_head_labels.append(head_labels)
                batch_tail_labels.append(tail_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_entity_labels = sequence_padding(
                        batch_entity_labels, seq_dims=2
                    )
                    batch_head_labels = sequence_padding(batch_head_labels, seq_dims=2)
                    batch_tail_labels = sequence_padding(batch_tail_labels, seq_dims=2)
                    yield {
                        "input_ids": torch.tensor(batch_token_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(
                            batch_segment_ids, dtype=torch.long
                        ),
                        "labels": [
                            torch.tensor(batch_entity_labels, dtype=torch.long),
                            torch.tensor(batch_head_labels, dtype=torch.long),
                            torch.tensor(batch_tail_labels, dtype=torch.long),
                        ],
                    }
                    batch_token_ids, batch_segment_ids = [], []
                    batch_entity_labels, batch_head_labels, batch_tail_labels = (
                        [],
                        [],
                        [],
                    )

    train_generator = data_generator(train_data, batch_size)

    @torch.no_grad()
    def extract_spoes(model, device, text, threshold=0):
        """抽取输入text所包含的三元组"""
        tokens = tokenizer.tokenize(text, maxlen=maxlen)
        mapping = tokenizer.rematch(text, tokens)
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)
        outputs = model(input_ids=token_ids, token_type_ids=segment_ids)
        outputs = [o[0].cpu().numpy() for o in outputs]
        # 抽取subject和object
        subjects, objects = set(), set()
        outputs[0][:, [0, -1]] -= np.inf
        outputs[0][:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(outputs[0] > threshold)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))
        # 识别对应的predicate
        spoes = set()
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add(
                        (
                            text[mapping[sh][0] : mapping[st][-1] + 1],
                            id2predicate[p],
                            text[mapping[oh][0] : mapping[ot][-1] + 1],
                        )
                    )
        return list(spoes)

    class SPO(tuple):
        """用来存三元组的类
        表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
        使得在判断两个三元组是否等价时容错性更好。
        """

        def __init__(self, spo):
            self.spox = (
                tuple(tokenizer.tokenize(spo[0])),
                spo[1],
                tuple(tokenizer.tokenize(spo[2])),
            )

        def __hash__(self):
            return self.spox.__hash__()

        def __eq__(self, spo):
            return self.spox == spo.spox

    def evaluate(model, device, data):
        """评估函数，计算f1、precision、recall"""
        X, Y, Z = 1e-10, 1e-10, 1e-10
        f = open("dev_pred.json", "w", encoding="utf-8")
        pbar = tqdm(
            desc="Evaluating: ",
        )
        model.eval()
        for d in data:
            R = set([SPO(spo) for spo in extract_spoes(model, device, d["text"])])
            T = set([SPO(spo) for spo in d["spo_list"]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            # pbar.set_description(
            #     "f1: %.5f, precision: %.5f, recall: %.5f" % (f1, precision, recall)
            # )
            s = json.dumps(
                {
                    "text": d["text"],
                    "spo_list": list(T),
                    "spo_list_pred": list(R),
                    "new": list(R - T),
                    "lack": list(T - R),
                },
                ensure_ascii=False,
                indent=4,
            )
            f.write(s + "\n")
        pbar.close()
        f.close()
        model.train()
        return f1, precision, recall


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    max_train_steps = epochs * len(train_generator)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps,
    )
    for epoch in range(1, epochs + 1):
        train(model, device, train_generator, lr_scheduler, optimizer)
        f1, precision, recall = evaluate(model, device, valid_data)
        print(
            f"#Epoch {epoch} -- f1 : {f1}, precision : {precision}, recall : {recall}"
        )
        print("=" * 50)


if __name__ == "__main__":
    main()
