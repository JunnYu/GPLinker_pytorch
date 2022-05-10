import numpy as np
import torch
import torch.nn as nn
from chinesebert.modeling_chinesebert import ChineseBertModel
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.roformer import RoFormerModel, RoFormerPreTrainedModel

model_name2model_cls = {
    "bert": (BertPreTrainedModel, BertModel),
    "chinesebert": (BertPreTrainedModel, ChineseBertModel),
    "roformer": (RoFormerPreTrainedModel, RoFormerModel),
}


INF = 1e4
EPSILON = 1e-5

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, unsqueeze_dim=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("n , d -> n d", t, inv_freq)
        if unsqueeze_dim:
            freqs = freqs.unsqueeze(unsqueeze_dim)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)

    def forward(self, t, seqlen=-2, past_key_value_length=0):
        # t shape [bs, dim, seqlen, seqlen]
        sin, cos = (
            self.sin[past_key_value_length : past_key_value_length + seqlen, :],
            self.cos[past_key_value_length : past_key_value_length + seqlen, :],
        )
        t1, t2 = t[..., 0::2], t[..., 1::2]
        # 奇偶交错
        return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(
            -2, -1
        )
        
class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, heads=12, head_size=64, RoPE=True, tril_mask=True, max_length=512):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * 2 * head_size)
        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length, unsqueeze_dim=-2) # n1d

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense(inputs)
        bs, seqlen = inputs.shape[:2]

        # method 1
        inputs = inputs.reshape(bs, seqlen, self.heads, 2, self.head_size)
        qw, kw = inputs.unbind(axis=-2)

        # method 2
        # inputs = inputs.reshape(bs, seqlen, self.heads, 2 * self.head_size)
        # qw, kw = inputs.chunk(2, axis=-1)

        # original
        # inputs = inputs.chunk(self.heads, axis=-1)
        # inputs = torch.stack(inputs, axis=-2)
        # qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]

        # RoPE编码
        if self.RoPE:
            qw, kw = self.rotary(qw, seqlen), self.rotary(kw, seqlen)

        # 计算内积
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * INF

        # 排除下三角
        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)

            logits = logits - mask * INF

        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(self, hidden_size, heads=12, head_size=64, RoPE=True, tril_mask=True, max_length=512):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, head_size * 2)
        self.dense2 = nn.Linear(head_size * 2, heads * 2)
        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length)

    def forward(self, inputs, attention_mask=None):
        seqlen = inputs.shape[1]
        inputs = self.dense1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            qw, kw = self.rotary(qw, seqlen), self.rotary(kw, seqlen)
            
        # 计算内积
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.head_size ** 0.5
        bias = self.dense2(inputs).transpose(1, 2) / 2  #'bnh->bhn'
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * INF

        # 排除下三角
        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)

            logits = logits - mask * INF

        return logits


def sparse_multilabel_categorical_crossentropy(
    y_true, y_pred, mask_zero=False
):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + INF
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=EPSILON, max=1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss


def globalpointer_loss(y_pred, y_true):
    shape = y_pred.shape
    # bs, nclass, max_spo_num
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    # bs, nclass, seqlen * seqlen
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=True)
    return loss.sum(dim=1).mean()


def get_auto_model(model_type):
    parent_cls, base_cls = model_name2model_cls[model_type]
    exist_add_pooler_layer = model_type in ["bert"]

    class AutoModelGPLinker4EE(parent_cls):
        def __init__(self, config, head_size=64, use_efficient=False):
            super().__init__(config)
            if exist_add_pooler_layer:
                setattr(
                    self,
                    self.base_model_prefix,
                    base_cls(config, add_pooling_layer=False),
                )
            else:
                setattr(self, self.base_model_prefix, base_cls(config))
            if use_efficient:
                gpcls = EfficientGlobalPointer
            else:
                gpcls = GlobalPointer
            self.argu_output = gpcls(
                hidden_size=config.hidden_size,
                heads=config.num_labels,
                head_size=head_size,
                tril_mask=True,
            )
            self.head_output = gpcls(
                hidden_size=config.hidden_size,
                heads=1,
                head_size=head_size,
                RoPE=False,
                tril_mask=True,
            )
            self.tail_output = gpcls(
                hidden_size=config.hidden_size,
                heads=1,
                head_size=head_size,
                RoPE=False,
                tril_mask=True,
            )
            self.post_init()

        def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
        ):

            outputs = getattr(self, self.base_model_prefix)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False,
                **kwargs,
            )
            last_hidden_state = outputs[0]
            argu_output = self.argu_output(
                last_hidden_state, attention_mask=attention_mask
            )
            head_output = self.head_output(
                last_hidden_state, attention_mask=attention_mask
            )
            tail_output = self.tail_output(
                last_hidden_state, attention_mask=attention_mask
            )

            aht_output = (argu_output, head_output, tail_output)
            loss = None
            if labels is not None:
                loss = (
                    sum([globalpointer_loss(o, l) for o, l in zip(aht_output, labels)])
                    / 3
                )
            output = (aht_output,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

    return AutoModelGPLinker4EE
