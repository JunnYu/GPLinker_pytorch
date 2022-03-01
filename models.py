import numpy as np
import torch
import torch.nn as nn
from chinesebert.modeling_chinesebert import ChineseBertModel
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.roformer import RoFormerModel, RoFormerPreTrainedModel

from utils.components import HandshakingKernel

model_name2model_cls = {
    "bert": (BertPreTrainedModel, BertModel),
    "chinesebert": (BertPreTrainedModel, ChineseBertModel),
    "roformer": (RoFormerPreTrainedModel, RoFormerModel),
}


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(
        self,
        hidden_size,
        heads=12,
        head_size=64,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * 2 *
                               head_size, bias=use_bias)

    def get_rotary_positions_embeddings(self, inputs, output_dim):
        position_ids = torch.arange(
            0, inputs.size(1), dtype=inputs.dtype, device=inputs.device
        )

        indices = torch.arange(
            0, output_dim // 2, dtype=inputs.dtype, device=inputs.device
        )
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum("n,d->nd", position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], axis=-1).flatten(
            1, 2
        )
        return embeddings[None, :, :]

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
            pos = self.get_rotary_positions_embeddings(inputs, self.head_size)
            cos_pos = torch.repeat_interleave(pos[..., None, 1::2], 2, axis=-1)
            sin_pos = torch.repeat_interleave(pos[..., None, ::2], 2, axis=-1)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]],
                              axis=-1).reshape_as(qw)

            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]],
                              axis=-1).reshape_as(kw)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] *
                attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * 1e12

        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits = logits - mask * 1e12

        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(
        self,
        hidden_size,
        heads=12,
        head_size=64,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.dense2 = nn.Linear(head_size * 2, heads * 2, bias=use_bias)

    def get_rotary_positions_embeddings(self, inputs, output_dim):
        position_ids = torch.arange(
            inputs.size(1), dtype=inputs.dtype, device=inputs.device
        )

        indices = torch.arange(
            output_dim // 2, dtype=inputs.dtype, device=inputs.device
        )
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum("n,d->nd", position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], axis=-1).flatten(
            1, 2
        )
        return embeddings[None, :, :]

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            pos = self.get_rotary_positions_embeddings(inputs, self.head_size)
            cos_pos = torch.repeat_interleave(pos[..., 1::2], 2, axis=-1)
            sin_pos = torch.repeat_interleave(pos[..., ::2], 2, axis=-1)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]],
                              axis=-1).reshape_as(qw)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]],
                              axis=-1).reshape_as(kw)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.head_size ** 0.5
        bias = self.dense2(inputs).transpose(1, 2) / 2  # 'bnh->bhn'
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] *
                attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * 1e12

        # 排除下三角
        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)

            logits = logits - mask * 1e12

        return logits


def sparse_multilabel_categorical_crossentropy(
    y_true, y_pred, mask_zero=False, epsilon=1e-7, Inf=1e12
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
        infs = zeros + Inf
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=epsilon, max=1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss


def globalpointer_loss(y_pred, y_true):
    shape = y_pred.shape
    # bs, nclass, max_spo_num
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    # bs, nclass, seqlen * seqlen
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    loss = sparse_multilabel_categorical_crossentropy(
        y_true, y_pred, mask_zero=True)
    return loss.sum(dim=1).mean()


class LossCalculator:
    def __init__(self, ghm):
        self.last_weights = None  # for exponential moving averaging
        self.ghm = ghm

    def GHM(self, gradient, bins=10, beta=0.9):
        """
        gradient_norm: gradient_norms of all examples in this batch; (batch_size, shaking_seq_len)
        """
        avg = torch.mean(gradient)
        std = torch.std(gradient) + 1e-12
        gradient_norm = torch.sigmoid(
            (gradient - avg) / std
        )  # normalization and pass through sigmoid to 0 ~ 1.

        min_, max_ = torch.min(gradient_norm), torch.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        gradient_norm = torch.clamp(
            gradient_norm, 0, 0.9999999
        )  # ensure elements in gradient_norm != 1.

        example_sum = torch.flatten(gradient_norm).size()[0]  # N

        # calculate weights
        current_weights = torch.zeros(bins).to(gradient.device)
        hits_vec = torch.zeros(bins).to(gradient.device)
        count_hits = 0  # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = torch.sum((gradient_norm <= bar)) - count_hits
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / \
                (hits.item() + example_sum / bins)

        if self.last_weights is None:
            self.last_weights = torch.ones(bins).to(
                gradient.device)  # init by ones
        current_weights = self.last_weights * \
            beta + (1 - beta) * current_weights
        self.last_weights = current_weights

        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins)).long()[:, :, None]
        weights_rp = current_weights[None, None, :].repeat(
            gradient_norm.size()[0], gradient_norm.size()[1], 1
        )
        weights4examples = torch.gather(
            weights_rp, -1, weight_pk_idx).squeeze(-1)
        weights4examples /= torch.sum(weights4examples)
        return weights4examples * gradient  # return weighted gradients

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        # -1 -> pos classes, 1 -> neg classes
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred oudtuts of pos classes
        y_pred_pos = (
            y_pred - (1 - y_true) * 1e12
        )  # mask the pred oudtuts of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])  # st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        if self.ghm:
            return (self.GHM(neg_loss + pos_loss, bins=1000)).sum()
        else:
            return (neg_loss + pos_loss).mean()

    def __call__(self, y_pred, y_true):
        return self.multilabel_categorical_crossentropy(y_pred, y_true)


def get_auto_model(model_type, method="tplinker_plus"):
    parent_cls, base_cls = model_name2model_cls[model_type]
    exist_add_pooler_layer = model_type in ["bert"]

    if method == "gplinker":

        class AutoModelGPLinker(parent_cls):
            def __init__(self, config, predicate2id, head_size=64, use_efficient=False):
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
                self.entity_output = gpcls(
                    hidden_size=config.hidden_size, heads=2, head_size=head_size
                )
                self.head_output = gpcls(
                    hidden_size=config.hidden_size,
                    heads=len(predicate2id),
                    head_size=head_size,
                    RoPE=False,
                    tril_mask=False,
                )
                self.tail_output = gpcls(
                    hidden_size=config.hidden_size,
                    heads=len(predicate2id),
                    head_size=head_size,
                    RoPE=False,
                    tril_mask=False,
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
                entity_output = self.entity_output(
                    last_hidden_state, attention_mask=attention_mask
                )
                head_output = self.head_output(
                    last_hidden_state, attention_mask=attention_mask
                )
                tail_output = self.tail_output(
                    last_hidden_state, attention_mask=attention_mask
                )

                spo_output = (entity_output, head_output, tail_output)
                loss = None
                if labels is not None:
                    loss = (
                        sum(
                            [
                                globalpointer_loss(o, l)
                                for o, l in zip(spo_output, labels)
                            ]
                        )
                        / 3
                    )
                output = (spo_output,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

        return AutoModelGPLinker

    elif method == "tplinker_plus":

        class AutoModelTPLinkerPlus(parent_cls):
            def __init__(
                self,
                config,
                predicate2id,
                shaking_type="cln",
                inner_enc_type="mix_pooling",
                ghm=False,
            ):
                super().__init__(config)
                if exist_add_pooler_layer:
                    setattr(
                        self,
                        self.base_model_prefix,
                        base_cls(config, add_pooling_layer=False),
                    )
                else:
                    setattr(self, self.base_model_prefix, base_cls(config))
                self.post_init()
                self.handshaking_kernel = HandshakingKernel(
                    config.hidden_size, shaking_type, inner_enc_type
                )
                self.out_proj = nn.Linear(
                    config.hidden_size, len(predicate2id) * 4 + 1)

                self.loss_fn = LossCalculator(ghm)

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
                # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
                shaking_hiddens = self.handshaking_kernel(last_hidden_state)

                # shaking_logits: (batch_size, shaking_seq_len, tag_size)
                shaking_logits = self.out_proj(shaking_hiddens)

                loss = None
                if labels is not None:
                    loss = self.loss_fn(shaking_logits, labels)

                output = (shaking_logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

        return AutoModelTPLinkerPlus
