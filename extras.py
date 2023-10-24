import math
import sys

import torch
from torch import nn
from typing import Optional, Tuple, List
from volta_src.embeddings import BertLayerNorm
from volta_src.encoders import ACT2FN


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer).float()
        key_layer = self.transpose_for_scores(mixed_key_layer).float()
        value_layer = self.transpose_for_scores(mixed_value_layer).float()

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = hidden_states
        hidden_states = self.dense(hidden_states.half())
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor.half(), attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.relu = torch.nn.ReLU()
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            input_type: bool = False,
            tau=1.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attention_mask, -10000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if input_type:
            attn_weights = 1.0 - torch.nn.functional.softmax(attn_weights / tau, dim=-1)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights / tau, dim=-1)
        # 需要调整这个参数以让模型选择自己想要的patch区域，
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        if config.hidden_size == 1024:
            self.cross_attention = BartAttention(config.hidden_size, 8, dropout=0.01, is_decoder=True)
            self.cross_vision = BartAttention(config.hidden_size, 8, dropout=0.01, is_decoder=True)
        else:
            self.cross_attention = BartAttention(config.hidden_size, 8, dropout=0.1, is_decoder=True)
            self.cross_vision = BartAttention(config.hidden_size, 8, dropout=0.1, is_decoder=True)
        self.intermediate = BertIntermediate(config)
        self.weiget_F = torch.nn.Linear(config.hidden_size * 2, 1, bias=True)
        self.output = BertOutput(config)

    def forward(self, hidden_states=None, attention_mask=None, encoder_out=None, image_out=None, inferer_out=None, input_type=False):
        if hidden_states is not None:
            attention_output = self.attention(hidden_states, attention_mask)
        if encoder_out is not None:
            cross_r = self.cross_attention(attention_output.half(), encoder_out, output_attentions=False)
            attention_output = cross_r[0] - attention_output
            # cross_score = cross_r[1]
            # cross_score = cross_score.view(cross_score.size(0), -1, cross_score.size(-1)).contiguous()
            # cross_score = torch.mean(cross_score, dim=-2, keepdim=True)
            #cross_score = nn.Softmax(dim=-1)(cross_score)
            # attention_output = torch.bmm(cross_score, encoder_out)
        if image_out is not None:
            #### conjunction
            cross_vision_F = self.cross_attention(hidden_states.transpose(0, 1), image_out[:, :image_out.size(1) // 2, :].half(), output_attentions=True, input_type=False, tau=1.0)
            cross_vision_T = self.cross_attention(hidden_states.transpose(0, 1), image_out[:, image_out.size(1) // 2:, :].half(), output_attentions=True, input_type=False, tau=1.0)
            weight_F = torch.sigmoid(self.weiget_F(torch.cat([hidden_states, cross_vision_F[0].transpose(0, 1)], dim=-1))) * cross_vision_F[0].transpose(0, 1)
            weight_T = torch.sigmoid(self.weiget_F(torch.cat([hidden_states, cross_vision_T[0].transpose(0, 1)], dim=-1))) * cross_vision_T[0].transpose(0, 1)
            attention_output = weight_T + hidden_states + weight_F
            #weight_F = self.weiget_F(torch.cat([hidden_states, cross_vision_F[0].transpose(0, 1)], dim=-1))
            #weight_T = self.weiget_F(torch.cat([hidden_states, cross_vision_T[0].transpose(0, 1)], dim=-1))
            #weight = nn.Softmax(dim=-1)(torch.cat([weight_T, weight_F], dim=-1) * 2)
            #attention_output = weight[:, :, :1]*cross_vision_T[0].transpose(0, 1) + hidden_states + weight[:, :, 1:] * cross_vision_F[0].transpose(0, 1)
            #### conjunction without negation module
            # cross_vision_T = self.cross_attention(hidden_states.transpose(0, 1), image_out.half(), output_attentions=True, input_type=False, tau=1.0)
            # attention_output = cross_vision_T[0].transpose(0, 1) + hidden_states
        if inferer_out is not None:
            # cross_r = self.cross_attention(hidden_states.half(), inferer_out[:, :inferer_out.size(1) // 2, :], output_attentions=False)
            # cross_w = self.cross_attention(hidden_states.half(), inferer_out[:, inferer_out.size(1) // 2:, :], output_attentions=False)
            # attention_output = hidden_states + cross_r[0] + cross_w[0]
            cross_r = self.cross_attention(hidden_states.half(), inferer_out.half(), output_attentions=False)
            attention_output = cross_r[0] + hidden_states
        intermediate_output = self.intermediate(attention_output.half())
        layer_output = self.output(intermediate_output.half(), attention_output)
        return layer_output

"""
The above modules are copied from BERT.
"""


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    tokenized_sentences = []

    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokenized_sentences.append(tokens)

    max_len = max(len(tokens) for tokens in tokenized_sentences)

    for (i, tokens) in enumerate(tokenized_sentences):
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


"""
The above modules are copied from LXMERT.
"""
