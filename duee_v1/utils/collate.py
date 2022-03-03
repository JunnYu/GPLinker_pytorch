import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from fastcore.all import patch_to
from transformers.file_utils import PaddingStrategy, _is_torch_device
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


ignore_list = ["offset_mapping", "text"]


@patch_to(BatchEncoding)
def to(self, device):
    if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
        data = {}
        for k, v in self.data.items():
            if k in ignore_list:
                data[k] = v
            else:
                if isinstance(v, (tuple, list)) and isinstance(v[0], dict):
                    data[k] = [
                        {subk: subv.to(device) for subk, subv in vv.items()} for vv in v
                    ]
                elif isinstance(v, (tuple, list)) and isinstance(v[0], torch.Tensor):
                    data[k] = [vv.to(device) for vv in v]
                else:
                    data[k] = v.to(device=device)
        self.data = data
    else:
        logger.warning(
            f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported."
        )
    return self


@dataclass
class DataCollatorForGPLinkerDuEE:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        new_features = [
            {k: v for k, v in f.items() if k not in ["labels"] + ignore_list}
            for f in features
        ]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["text"] = [feature["text"] for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [
                    feature["offset_mapping"] for feature in features
                ]
            return batch

        bs = batch["input_ids"].size(0)
        max_head_num = max([len(lb["head_labels"]) for lb in labels])
        max_tail_num = max([len(lb["tail_labels"]) for lb in labels])
        max_argu_num = max(
            [(len(lb) - 1) // 2 for label in labels for lb in label["argu_labels"]]
        )
        batch_argu_labels = torch.zeros(
            bs, self.num_labels, max_argu_num * 2, dtype=torch.long
        )
        batch_head_labels = torch.zeros(bs, 1, max_head_num, 2, dtype=torch.long)
        batch_tail_labels = torch.zeros(bs, 1, max_tail_num, 2, dtype=torch.long)

        for b, lb in enumerate(labels):
            # argu_labels
            for argu in lb["argu_labels"]:
                batch_argu_labels[b, argu[0], : len(argu[1:])] = torch.tensor(argu[1:], dtype=torch.long)
            # head_labels
            for ih, (h1, h2) in enumerate(lb["head_labels"]):
                batch_head_labels[b, 0, ih, :] = torch.tensor([h1, h2], dtype=torch.long)
            # tail_labels
            for it, (t1, t2) in enumerate(lb["tail_labels"]):
                batch_tail_labels[b, 0, it, :] = torch.tensor([t1, t2], dtype=torch.long)

        batch["labels"] = [
            batch_argu_labels.reshape(bs, self.num_labels, max_argu_num, 2),
            batch_head_labels,
            batch_tail_labels,
        ]
        return batch
