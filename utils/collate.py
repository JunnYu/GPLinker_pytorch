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
class DataCollatorForGPLinker:
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
        max_spo_num = max([len(lb) for lb in labels])
        batch_entity_labels = torch.zeros(bs, 2, max_spo_num, 2, dtype=torch.long)
        batch_head_labels = torch.zeros(
            bs, self.num_labels, max_spo_num, 2, dtype=torch.long
        )
        batch_tail_labels = torch.zeros(
            bs, self.num_labels, max_spo_num, 2, dtype=torch.long
        )
        for i, lb in enumerate(labels):
            for spidx, (sh, st, p, oh, ot) in enumerate(lb):
                batch_entity_labels[i, 0, spidx, :] = torch.tensor([sh, st])
                batch_entity_labels[i, 1, spidx, :] = torch.tensor([oh, ot])
                batch_head_labels[i, p, spidx, :] = torch.tensor([sh, oh])
                batch_tail_labels[i, p, spidx, :] = torch.tensor([st, ot])

        batch["labels"] = [batch_entity_labels, batch_head_labels, batch_tail_labels]
        return batch


@dataclass
class DataCollatorForTPLinkerPlus:
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
        seqlen = batch["input_ids"].size(1)
        mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=0).bool()

        num_tag = self.num_labels * 4 + 1
        batch_shaking_tag = torch.zeros(bs, seqlen, seqlen, num_tag, dtype=torch.long)

        #  labels [#batch 1->[[left1,right1,type1],[left2,right2,type2]], #batch2->[[left1,right1,type1],[left2,right2,type2]]]

        for i, lb in enumerate(labels):
            for sh, st, p, oh, ot in lb:
                # SH2OH
                batch_shaking_tag[i, sh, oh, p] = 1
                # OH2SH
                batch_shaking_tag[i, oh, sh, p + self.num_labels] = 1
                # ST2OT
                batch_shaking_tag[i, st, ot, p + self.num_labels * 2] = 1
                # OT2ST
                batch_shaking_tag[i, ot, st, p + self.num_labels * 3] = 1
                # EH2ET
                batch_shaking_tag[i, sh, st, -1] = 1
                batch_shaking_tag[i, oh, ot, -1] = 1

        batch["labels"] = batch_shaking_tag.masked_select(
            mask[None, :, :, None]
        ).reshape(bs, -1, num_tag)
        return batch


# @dataclass
# class DataCollatorForTPLinkerPlus:
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     num_labels: Optional[int] = None

#     def __call__(
#         self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
#     ) -> Dict[str, torch.Tensor]:
#         labels = (
#             [feature["labels"] for feature in features]
#             if "labels" in features[0].keys()
#             else None
#         )
#         new_features = [{k: v for k, v in f.items() if k != "labels"}
#                         for f in features]
#         batch = self.tokenizer.pad(
#             new_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         if labels is None:
#             return batch

#         bs = batch["input_ids"].size(0)
#         seqlen = batch["input_ids"].size(1)
#         mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=0).long()
#         max_handshaking_num = seqlen * (seqlen + 1) // 2
#         matrix_idx2shaking_idx = mask.masked_scatter(
#             mask, torch.arange(max_handshaking_num)

#         batch_shaking_tag = torch.zeros(
#             bs, max_handshaking_num, self.num_labels, dtype=torch.long)

#         #  labels [#batch 1->[[left1,right1,type1],[left2,right2,type2]], #batch2->[[left1,right1,type1],[left2,right2,type2]]]

#         num_relations = (self.num_labels - 1) // 4

#         for i, lb in enumerate(labels):
#             for sh, st, p, oh, ot in lb:
#                 # SH2OH
#                 sh2oh = matrix_idx2shaking_idx[sh, oh]
#                 batch_shaking_tag[i, sh2oh, p] = 1
#                 # OH2SH
#                 oh2sh = matrix_idx2shaking_idx[oh, sh]
#                 batch_shaking_tag[i, oh2sh, p+num_relations] = 1
#                 # ST2OT
#                 st2ot = matrix_idx2shaking_idx[st, ot]
#                 batch_shaking_tag[i, st2ot, p+num_relations*2] = 1
#                 # OT2ST
#                 ot2st = matrix_idx2shaking_idx[ot, st]
#                 batch_shaking_tag[i, ot2st, p+num_relations*3] = 1
#                 # EH2ET
#                 seh2et = matrix_idx2shaking_idx[sh, st]
#                 oeh2et = matrix_idx2shaking_idx[oh, ot]
#                 batch_shaking_tag[i, seh2et, self.num_labels] = 1
#                 batch_shaking_tag[i, oeh2et, self.num_labels] = 1

#         batch["labels"] = batch_shaking_tag
#         return batch
