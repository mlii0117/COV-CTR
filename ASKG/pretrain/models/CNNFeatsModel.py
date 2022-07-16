from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import copy
import json
import logging
import math
import os
import shutil
import numpy as np
import tarfile
import tempfile
import sys
from io import open

import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.nn.parameter import Parameter

from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {"openai-gpt": "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-pytorch_model.bin"}
PRETRAINED_CONFIG_ARCHIVE_MAP = {"openai-gpt": "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-config.json"}

CONFIG_NAME = "transformer-config.json"
WEIGHTS_NAME = "pytorch_model.bin"


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)



ACT_FNS = {"relu": nn.ReLU, "swish": swish, "gelu": gelu}


class OpenAIGPTConfig(object):
    """Configuration class to store the configuration of a `OpenAIGPTModel`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file=40478,
        n_special=0,
        n_positions=512,
        n_ctx=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        afn="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        att_feat_size=1024,
        n_ann=49,
    ):
        """Constructs OpenAIGPTConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `OpenAIGPTModel` or a configuration json file.
            n_special: The number of special tokens to learn during fine-tuning ('[SEP]', '[CLF]', ...)
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            afn: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            resid_pdrop: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attn_pdrop: The dropout ratio for the attention
                probabilities.
            embd_pdrop: The dropout ratio for the embeddings.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.n_special = n_special
            self.n_ctx = n_ctx
            self.n_positions = n_positions
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.afn = afn
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.attn_pdrop = attn_pdrop
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_range = initializer_range
            self.att_feat_size = att_feat_size
            self.n_ann = n_ann
            self.tag_decoderdir = tag_decoderdir
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @property
    def total_tokens_embeddings(self):
        return self.vocab_size + self.n_special

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `OpenAIGPTConfig` from a Python dictionary of parameters."""
        config = OpenAIGPTConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `OpenAIGPTConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.weight = Parameter(w)
            self.bias = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class SelfAttention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, mask=False, weight=None):
        super(SelfAttention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.mask = mask
        self.weight = weight
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        # TODO b = self.bias[:, :, : w.size(-2), : w.size(-1)]
        if self.mask:
            b = self.bias[:, :, : w.size(-2), : w.size(-1)]
            w = w * b + -1e9 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        if self.weight is not None:
            w = w * self.weight
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class EncoderAttention(nn.Module):
    def __init__(self, nx, config, scale=False):
        super(EncoderAttention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0

        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn_q = Conv1D(n_state, 1, nx)
        self.c_attn_k = Conv1D(n_state, 1, nx)
        self.c_attn_v = Conv1D(n_state, 1, nx)

        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, src):
        '''
        :param x: decoded tokens
        :param src: img_feats
        '''
        # x = self.c_attn(x)
        # query, key, value = x.split(self.split_size, dim=2)
        query = self.c_attn_q(x)
        key = self.c_attn_k(src)
        value = self.c_attn_v(src)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class EncoderBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(EncoderBlock, self).__init__()

        nx = config.n_embd
        self.attn = EncoderAttention(nx, config, scale)
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)

        self.attn2 = SelfAttention(nx, n_ctx, config, scale, mask=False)
        self.ln_3 = LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, src):
        '''
        :param x: decoded tokens
        :param src: img_feats
        :return:
        '''

        a = self.attn(x, src)
        n = self.ln_1(x + a)

        a2 = self.attn2(n)
        n2 = self.ln_3(n + a2)

        # m = self.mlp(n2)
        # h = self.ln_2(n2 + m)
        # return h
        return n2


class DecoderBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(DecoderBlock, self).__init__()
        nx = config.n_embd
        self.attn = SelfAttention(nx, n_ctx, config, scale, mask=True)
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)

        # Add encoder multi-head branch
        self.attn2 = EncoderAttention(nx, config, scale)
        self.ln_3 = LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, src):
        '''
        :param x: decoded tokens
        :param src: img_feats
        :return:
        '''

        a = self.attn(x)
        n = self.ln_1(x + a)

        a2 = self.attn2(n, src)
        n2 = self.ln_3(n + a2)

        m = self.mlp(n2)
        h = self.ln_2(n2 + m)
        return h


class TagHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, config):
        super(TagHead, self).__init__()
        self.n_embd = config.n_embd

        self.decoder = nn.Sequential(
            nn.Linear(self.n_embd, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits.squeeze(2)


class LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, config):
        super(LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)

        # TODO
        # return F.log_softmax(lm_logits, dim=-1)
        return lm_logits


class OpenAIGPTPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(OpenAIGPTPreTrainedModel, self).__init__()
        if not isinstance(config, OpenAIGPTConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `OpenAIGPTConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_num_special_tokens(self, num_special_tokens):
        pass

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, num_special_tokens=None, state_dict=None, cache_dir=None, from_tf=False, *inputs, **kwargs
    ):
        """
        Instantiate a OpenAIGPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `openai-gpt`
                - a path or url to a pretrained model archive containing:
                    . `openai_gpt_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a OpenAIGPTModel instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . a series of NumPy files containing OpenAI TensorFlow trained weights
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find files {} and {} "
                "at this path or url.".format(
                    pretrained_model_name_or_path, ", ".join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), pretrained_model_name_or_path,
                    archive_file, config_file
                )
            )
            return None
        if resolved_archive_file == archive_file and resolved_config_file == config_file:
            logger.info("loading weights file {}".format(archive_file))
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))
        # Load config
        config = OpenAIGPTConfig.from_json_file(resolved_config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu' if not torch.cuda.is_available() else None)
        # if from_tf:
        #     # Directly load from a TensorFlow checkpoint (stored as NumPy array)
        #     return load_tf_weights_in_openai_gpt(model, resolved_archive_file)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_model = model
        if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
            start_model = model.transformer
        load(start_model, prefix="")

        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys)
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
            )

        # Add additional embeddings for special tokens if needed
        # This step also make sure we are still sharing the output and input embeddings after loading weights
        model.set_num_special_tokens(num_special_tokens if num_special_tokens is not None else config.n_special)
        return model


class ImageEncoderModel(OpenAIGPTPreTrainedModel):
    def __init__(self, model_embedding_weights, tag_decoder, config):
        super(ImageEncoderModel, self).__init__(config)
        self.tag_decoder = tag_decoder

        self.drop = nn.Dropout(config.embd_pdrop)
        block = EncoderBlock(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_elayer)])
        self.apply(self.init_weights)

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.set_embeddings_weights(model_embedding_weights)
        # nn.init.normal_(self.embed.weight, std=0.02)
    
    def set_embeddings_weights(self, model_embedding_weights):
        self.tokens_embed.weight = model_embedding_weights

    def set_num_special_tokens(self, num_special_tokens):
        " Update input embeddings with new embedding matrice if needed "
        if self.config.n_special == num_special_tokens:
            return
        # Update config
        self.config.n_special = num_special_tokens
        # # Build new embeddings and initialize
        old_embed = self.tokens_embed
        self.tokens_embed = nn.Embedding(self.config.total_tokens_embeddings, self.config.n_embd)
        # Initialize all new embeddings (in particular the special tokens)
        self.init_weights(self.tokens_embed)
        # Copy word and positional embeddings from the previous weights
        self.tokens_embed.weight.data[: self.config.vocab_size, :] = old_embed.weight.data[: self.config.vocab_size, :]
        self.tokens_embed.weight.data[-self.config.n_positions :, :] = old_embed.weight.data[-self.config.n_positions :, :]

    def forward(self, src):
        tag_ids = np.array([v for v in self.tag_decoder.values()])
        # tag_ids = torch.arange(self.config.num_medterms, dtype=torch.long, device=src.device)
        tag_ids = torch.from_numpy(tag_ids).to(src.device)
        tag_ids = tag_ids.unsqueeze(0).expand(src.size(0), -1)
        input_shape = tag_ids.size()
        tag_feats = self.tokens_embed(tag_ids)
        hidden_states = tag_feats
        for block in self.h:
            hidden_states = block(hidden_states, src)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape)




class SentenceEncoderModel(OpenAIGPTPreTrainedModel):
    def __init__(self, model_embedding_weights, tag_decoder, config):
        super(SentenceEncoderModel, self).__init__(config)
        self.tag_decoder = tag_decoder

        self.drop = nn.Dropout(config.embd_pdrop)
        block = EncoderBlock(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_elayer)])
        self.apply(self.init_weights)

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.set_embeddings_weights(model_embedding_weights)
        # nn.init.normal_(self.embed.weight, std=0.02)
    
    def set_embeddings_weights(self, model_embedding_weights):
        self.tokens_embed.weight = model_embedding_weights

    def set_num_special_tokens(self, num_special_tokens):
        " Update input embeddings with new embedding matrice if needed "
        if self.config.n_special == num_special_tokens:
            return
        # Update config
        self.config.n_special = num_special_tokens
        # # Build new embeddings and initialize
        old_embed = self.tokens_embed
        self.tokens_embed = nn.Embedding(self.config.total_tokens_embeddings, self.config.n_embd)
        # Initialize all new embeddings (in particular the special tokens)
        self.init_weights(self.tokens_embed)
        # Copy word and positional embeddings from the previous weights
        self.tokens_embed.weight.data[: self.config.vocab_size, :] = old_embed.weight.data[: self.config.vocab_size, :]
        self.tokens_embed.weight.data[-self.config.n_positions :, :] = old_embed.weight.data[-self.config.n_positions :, :]

    def forward(self, src):
        tag_ids = np.array([v for v in self.tag_decoder.values()])
        # tag_ids = torch.arange(self.config.num_medterms, dtype=torch.long, device=src.device)
        tag_ids = torch.from_numpy(tag_ids).to(src.device)
        tag_ids = tag_ids.unsqueeze(0).expand(src.size(0), -1)
        input_shape = tag_ids.size()
        tag_feats = self.tokens_embed(tag_ids)
        hidden_states = tag_feats
        for block in self.h:
            hidden_states = block(hidden_states, src)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape)


class DecoderModel(OpenAIGPTPreTrainedModel):
    def __init__(self, model_embedding_weights, config):
        super(DecoderModel, self).__init__(config)
        num_tokens = config.vocab_size + config.n_special
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        block = DecoderBlock(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_dlayer)])

        self.apply(self.init_weights)

        self.tokens_embed = nn.Embedding(num_tokens, config.n_embd)
        self.set_embeddings_weights(model_embedding_weights)
        # nn.init.normal_(self.embed.weight, std=0.02)

    def set_embeddings_weights(self, model_embedding_weights):
        self.tokens_embed.weight = model_embedding_weights

    def set_num_special_tokens(self, num_special_tokens):
        " Update input embeddings with new embedding matrice if needed "
        if self.config.n_special == num_special_tokens:
            return
        # Update config
        self.config.n_special = num_special_tokens
        # # Build new embeddings and initialize
        old_embed = self.tokens_embed
        self.tokens_embed = nn.Embedding(self.config.total_tokens_embeddings, self.config.n_embd)
        # Initialize all new embeddings (in particular the special tokens)
        self.init_weights(self.tokens_embed)
        # Copy word and positional embeddings from the previous weights
        self.tokens_embed.weight.data[: self.config.vocab_size, :] = old_embed.weight.data[: self.config.vocab_size, :]
        self.tokens_embed.weight.data[-self.config.n_positions :, :] = old_embed.weight.data[-self.config.n_positions :, :]

    def forward(self, input_ids, position_ids=None, token_type_ids=None, src=None):
        if position_ids is None:
            # This was used when we had a single embedding matrice from position and token embeddings
            # start = self.config.vocab_size + self.config.n_special
            # end = start + input_ids.size(-1)
            # position_ids = torch.arange(start, end, dtype=torch.long, device=input_ids.device)
            position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        # Add the position information to the input embeddings
        # h = e.sum(dim=2)
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        for block in self.h:
            hidden_states = block(hidden_states, src)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape)


class SentenceLMHeadModel(OpenAIGPTPreTrainedModel):
    def __init__(self, tag_decoder, config):
        super(SentenceLMHeadModel, self).__init__(config)
        self.config = config

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)

        self.stencoder = SentenceEncoderModel(self.tokens_embed.weight, tag_decoder, config)
        self.encoder = ImageEncoderModel(self.tokens_embed.weight, tag_decoder, config)
        self.decoder = DecoderModel(self.tokens_embed.weight, config)
        self.tag_head = TagHead(config)
        self.lm_head = LMHead(self.decoder.tokens_embed.weight, config)
        self.apply(self.init_weights)
        # self.set_num_special_tokens(config.n_special)

        # TODO
        self.img_feat_size = config.img_feat_size
        self.img_ann = config.img_ann
        self.att_feat_size = config.att_feat_size
        self.n_ann = config.n_ann
        self.seq_sent_length = config.seq_sent_length
        self.d_model = config.n_embd

        self.ss_prob = 0.0  # Schedule sampling probability

        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.d_model),
                                       nn.ReLU())
        self.img_att_embed = nn.Sequential(nn.Linear(self.img_feat_size, self.d_model),
                                       nn.ReLU())

    def set_num_special_tokens(self, num_special_tokens):
        """ Update input and output embeddings with new embedding matrice
            Make sure we are sharing the embeddings
        """
        self.decoder.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.decoder.tokens_embed.weight)

    def forward(self, att_feats, input_ids=None, position_ids=None, token_type_ids=None, mode='train', input_type='sentence'):
        if mode == 'train':
            return self._forward(att_feats, input_ids, position_ids, token_type_ids, input_type)
        else:
            return self._sample(att_feats)

    def _forward(self, att_feats, input_ids, position_ids=None, token_type_ids=None, input_type='sentence'):
        if input_type == 'sentence':
            att_feats = self.tokens_embed(att_feats)
            batch_size = att_feats.size(0)
            att_feats = att_feats.view(batch_size, self.n_ann, self.att_feat_size)
            att_feats = self.att_embed(att_feats)
            tag_hidden_states = self.stencoder(att_feats)
        if input_type == 'img':
            batch_size = att_feats.size(0)
            att_feats = att_feats.view(batch_size, self.img_ann, self.img_feat_size)
            att_feats = self.img_att_embed(att_feats)
        hidden_states = self.decoder(input_ids, position_ids, token_type_ids, att_feats)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits

    def _sample(self, att_feats):
        # 40478 start_token || 40479 end_token
        start_token = 2
        end_token = 3
        batch_size = att_feats.size(0)
        att_feats = att_feats.view(batch_size, self.img_ann, self.img_feat_size)
        att_feats = self.img_att_embed(att_feats)
        unfinish = torch.zeros(batch_size).long().cuda()
        ys = torch.ones(batch_size, 1).fill_(start_token).long().cuda()  # start token = 0

        for i in range(self.seq_sent_length - 1):
            hidden_states = self.decoder(ys, None, None, att_feats)
            prob = self.lm_head(hidden_states[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            mask = (next_word == end_token)
            unfinish[mask] = 1
            if unfinish.sum() == batch_size:
                break

        return ys

