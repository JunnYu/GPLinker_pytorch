import torch
import torch.nn as nn
from transformers.models.bert import BertModel, BertPreTrainedModel

# def globalpointer_loss(y_pred, y_true):
#     n2 = y_pred.size(-1) ** 2
#     y_pred = y_pred.reshape(-1, n2)
#     y_true = y_true.reshape(-1, n2)
#     y_true = y_true.to(y_pred.dtype)
#     y_pred = (1 - 2 * y_true) * y_pred
#     y_pred_neg = y_pred - y_true * 1e12
#     y_pred_pos = y_pred - (1 - y_true) * 1e12
#     zeros = torch.zeros_like(y_pred[..., :1])
#     y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
#     y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
#     neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
#     pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
#     return (neg_loss + pos_loss).mean()


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
        self.dense = nn.Linear(hidden_size, heads * 2 * head_size, bias=use_bias)

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

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], axis=-1).reshape_as(qw)

            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], axis=-1).reshape_as(kw)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
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

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], axis=-1).reshape_as(qw)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], axis=-1).reshape_as(kw)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.head_size ** 0.5
        bias = self.dense2(inputs).transpose(1, 2) / 2  #'bnh->bhn'
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * 1e12

        # 排除下三角
        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)

            logits = logits - mask * 1e12

        return logits


class GPLinker(BertPreTrainedModel):
    def __init__(self, config, predicate2id, efficient=False):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        if efficient:
            gpcls = EfficientGlobalPointer
        else:
            gpcls = GlobalPointer
        self.entity_output = gpcls(
            hidden_size=config.hidden_size, heads=2, head_size=64
        )
        self.head_output = gpcls(
            hidden_size=config.hidden_size,
            heads=len(predicate2id),
            head_size=64,
            RoPE=False,
            tril_mask=False,
        )
        self.tail_output = gpcls(
            hidden_size=config.hidden_size,
            heads=len(predicate2id),
            head_size=64,
            RoPE=False,
            tril_mask=False,
        )
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        last_hidden_state = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )[0]
        entity_output = self.entity_output(
            last_hidden_state, attention_mask=attention_mask
        )
        head_output = self.head_output(last_hidden_state, attention_mask=attention_mask)
        tail_output = self.tail_output(last_hidden_state, attention_mask=attention_mask)
        return entity_output, head_output, tail_output
