# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from fairseq.modules.knn_datastore import KNN_Dstore
import torch.nn.functional as functional
from torch_scatter import scatter
import torch.nn.functional as F

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("transformer")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    def load_state_dict(self, state_dict, strict=True, args=None):
        """we rewrite the load state dict here for only load part of trained model
        add by
        """
        if self.decoder.knn_lambda_type == 'trainable' or self.decoder.knn_temperature_type == 'trainable' \
                or self.decoder.use_knn_datastore \
                or (hasattr(self.encoder, "append_adapter") and self.encoder.append_adapter) \
                or (hasattr(self.encoder, "embedding_append_adapter") and self.encoder.embedding_append_adapter) \
                or (hasattr(self.decoder, "append_adapter") and self.decoder.append_adapter) \
                or (hasattr(self.encoder, "embedding_append_adapter") and self.encoder.embedding_append_adapter):

            self.upgrade_state_dict(state_dict)
            from fairseq.checkpoint_utils import prune_state_dict
            new_state_dict = prune_state_dict(state_dict, args)

            print('-----------------knn (adapter) load part of model-----------------')
            model_dict = self.state_dict()

            remove_keys = []
            for k, v in new_state_dict.items():
                if k not in model_dict:
                    remove_keys.append(k)

            for k in remove_keys:
                new_state_dict.pop(k)

            model_dict.update(new_state_dict)
            return super().load_state_dict(model_dict)

        else:
            return super().load_state_dict(state_dict, strict, args)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on
        # args for use knn and Training knn parameters

        parser.add_argument("--load-knn-datastore",
                            default=False, action='store_true')
        parser.add_argument("--dstore-filename", default=None, type=str)
        parser.add_argument("--use-knn-datastore",
                            default=False, action='store_true')

        parser.add_argument(
            "--dstore-fp16", action='store_true', help="if save only fp16")
        parser.add_argument("--dstore-size", metavar="N",
                            default=1, type=int, help="datastore size")
        parser.add_argument("--k", default=8, type=int)
        parser.add_argument("--probe", default=32, type=int)

        parser.add_argument("--faiss-metric-type", default=None, type=str)
        parser.add_argument("--knn-sim-func", default=None, type=str)

        parser.add_argument("--use-gpu-to-search",
                            default=False, action="store_true")
        parser.add_argument("--no-load-keys", default=False,
                            action="store_true")
        parser.add_argument("--move-dstore-to-mem",
                            default=False, action="store_true")
        parser.add_argument("--only-use-max-idx",
                            default=False, action="store_true")

        parser.add_argument("--knn-lambda-type", default="fix", type=str)
        parser.add_argument("--knn-lambda-value", default=0.5, type=float)
        parser.add_argument("--knn-lambda-net-hid-size", default=0, type=int)

        parser.add_argument("--label-count-as-feature",
                            default=False, action="store_true")
        parser.add_argument("--relative-label-count",
                            default=False, action="store_true")
        parser.add_argument("--knn-net-dropout-rate", default=0.5, type=float)

        #
        # parser.add_argument("--knn-lambda-net-input-label-count", default=)
        parser.add_argument("--knn-temperature-type", default="fix", type=str)
        parser.add_argument("--knn-temperature-value", default=10, type=float)
        parser.add_argument(
            "--knn-temperature-net-hid-size", default=0, type=int)

        # we add 4 arguments for trainable k network
        parser.add_argument("--knn-k-type", default="fix", type=str)
        parser.add_argument("--max-k", default=None, type=int)
        parser.add_argument("--knn-k-net-hid-size", default=0, type=int)
        parser.add_argument("--knn-k-net-dropout-rate", default=0, type=float)

        # we add 3 arguments for trainable k_with_lambda network
        parser.add_argument("--k-lambda-net-hid-size", type=int, default=0)
        parser.add_argument("--k-lambda-net-dropout-rate",
                            type=float, default=0.0)
        parser.add_argument("--gumbel-softmax-temperature",
                            type=float, default=1)

        parser.add_argument("--avg-k", default=False, action='store_true')

        parser.add_argument("--only-train-knn-parameter",
                            default=False, action='store_true')

        # we add arguments for injecting knn knowledge 
        parser.add_argument("--query-knn-datastore-during-training",
                            default=False, action='store_true')
        parser.add_argument("--knn-loss-type", default=None, type=str)
        parser.add_argument("--knn-loss-weight", default=0, type=float)
        parser.add_argument("--kd-loss-weight", default=0, type=float)

        # we add arguments for adapter
        parser.add_argument("--activate-adapter", default=False, action="store_true")
        parser.add_argument("--encoder-embedding-append-adapter", action="store_true", default=False)
        parser.add_argument("--decoder-embedding-append-adapter", action="store_true", default=False)
        parser.add_argument("--encoder-adapter-use-last-ln", action="store_true", default=False)
        parser.add_argument("--decoder-adapter-use-last-ln", action="store_true", default=False)
        parser.add_argument("--encoder-append-adapter", action="store_true", default=False)
        parser.add_argument("--decoder-append-adapter", action="store_true", default=False)
        parser.add_argument("--only-update-adapter", action="store_true", default=False)
        parser.add_argument("--adapter-ffn-dim", type=int, default=0)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError(
                    "--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        # we disable the parameter updated other than knn parameter
        if args.only_train_knn_parameter:

            for name, param in encoder.named_parameters():
                param.requires_grad = False

            for name, param in decoder.named_parameters():
                param.requires_grad = False

            for name, param in decoder.named_parameters():

                if "knn_distance_to_lambda" in name and decoder.knn_lambda_type == "trainable":
                    param.requires_grad = True

                if "knn_distance_to_k" in name and decoder.knn_k_type == "trainable":
                    param.requires_grad = True

                if "retrieve_result_to_k_and_lambda" in name and decoder.knn_lambda_type == "trainable" \
                        and decoder.knn_k_type == "trainable":
                    param.requires_grad = True

        # we disable the parameter updated other than adapter parameter
        if args.only_update_adapter:

            for name, param in encoder.named_parameters():
                param.requires_grad = False

            for name, param in decoder.named_parameters():
                param.requires_grad = False

            for name, param in encoder.named_parameters():
                if "adapter" in name:
                    param.requires_grad = True

            for name, param in decoder.named_parameters():
                if "adapter" in name:
                    param.requires_grad = True

        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args)
             for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.append_adapter = args.encoder_append_adapter

        self.embedding_append_adapter = args.encoder_embedding_append_adapter
        if self.embedding_append_adapter:
            # self.normalize_before = args.encoder_normalize_before
            self.embedding_adapter_fc1 = nn.Linear(embed_dim, args.adapter_ffn_dim)
            self.embedding_adapter_fc2 = nn.Linear(args.adapter_ffn_dim, embed_dim)
            self.embedding_adapter_layer_norm = LayerNorm(embed_dim)
            self.embedding_adapter_dropout_module = FairseqDropout(
                float(args.dropout), module_name=self.__class__.__name__
            )
            self.embedding_adapter_activation_fn = utils.get_activation_fn(
                activation=getattr(args, "activation_fn", "relu")
            )
        
        self.activate_adapter = args.activate_adapter

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(
            src_tokens, token_embeddings)
        
        if self.activate_adapter and self.embedding_append_adapter:

            residual = x
            # if self.normalize_before:
            x = self.embedding_adapter_layer_norm(x)

            x = self.embedding_adapter_activation_fn(self.embedding_adapter_fc1(x))
            x = self.embedding_adapter_dropout_module(x)
            x = self.embedding_adapter_fc2(x)
            x = self.embedding_adapter_dropout_module(x)
            x = residual + x

            # if not self.normalize_before:
            #     x = self.embedding_adapter_layer_norm(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask, activate_adapter=self.activate_adapter)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(
            args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

        # add by zhuwh
        self.args.only_return_mt_knn_results = False

        # add by  , trainable (distances to lambda and temperature) knn datastore
        self.fp16 = args.fp16
        self.knn_datastore = None
        if torch.cuda.device_count() > 1:
            args.use_all_gpus_to_search = True
        else:
            args.use_all_gpus_to_search = False
        if args.load_knn_datastore:
            self.knn_datastore = KNN_Dstore(args, len(dictionary))

        self.use_knn_datastore = args.use_knn_datastore
        self.knn_lambda_type = args.knn_lambda_type
        self.knn_temperature_type = args.knn_temperature_type
        self.knn_k_type = args.knn_k_type
        self.label_count_as_feature = args.label_count_as_feature
        self.relative_label_count = args.relative_label_count
        self.avg_k = args.avg_k

        # add by zhuwh
        self.query_knn_datastore_during_training = args.query_knn_datastore_during_training
        self.compute_knn_loss = False

        if self.knn_lambda_type == "trainable" and self.knn_k_type == "trainable":

            # TODO another network to predict k and lambda at the same time without gumbel softmax
            self.retrieve_result_to_k_and_lambda = nn.Sequential(
                nn.Linear(args.max_k if not self.label_count_as_feature else args.max_k * 2,
                          args.k_lambda_net_hid_size),
                nn.Tanh(),
                nn.Dropout(p=args.k_lambda_net_dropout_rate),
                nn.Linear(args.k_lambda_net_hid_size, 2 +
                          int(math.log(args.max_k, 2))),
                # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 4 , 8 , ... , ]
                nn.Softmax(dim=-1),
            )

            nn.init.xavier_normal_(
                self.retrieve_result_to_k_and_lambda[0].weight[:, : args.k], gain=0.01)

            if self.label_count_as_feature:
                nn.init.xavier_normal_(
                    self.retrieve_result_to_k_and_lambda[0].weight[:, args.k:], gain=0.1)

        else:
            if self.knn_lambda_type == 'trainable':
                # TODO, we may update the label count feature here
                self.knn_distances_to_lambda = nn.Sequential(
                    nn.Linear(args.k if not self.label_count_as_feature else args.k *
                              2, args.knn_lambda_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=args.knn_net_dropout_rate),
                    nn.Linear(args.knn_lambda_net_hid_size, 1),
                    nn.Sigmoid())

                if self.label_count_as_feature:
                    # nn.init.normal_(self.knn_distances_to_lambda[0].weight[:, :args.k], mean=0, std=0.01)
                    # nn.init.normal_(self.knn_distances_to_lambda[0].weight[:, args.k:], mean=0, std=0.1)

                    nn.init.xavier_normal_(
                        self.knn_distances_to_lambda[0].weight[:, : args.k], gain=0.01)
                    nn.init.xavier_normal_(
                        self.knn_distances_to_lambda[0].weight[:, args.k:], gain=0.1)
                    nn.init.xavier_normal_(
                        self.knn_distances_to_lambda[-2].weight)

                else:
                    nn.init.normal_(
                        self.knn_distances_to_lambda[0].weight, mean=0, std=0.01)

            if self.knn_temperature_type == 'trainable':
                # TODO, consider a reasonable function
                self.knn_distance_to_temperature = nn.Sequential(
                    nn.Linear(args.k + 2, args.knn_temperature_net_hid_size),
                    nn.Tanh(),
                    nn.Linear(args.knn_temperature_net_hid_size, 1),
                    nn.Sigmoid())
                # the weight shape is [net hid size, k + 1)
                nn.init.normal_(
                    self.knn_distance_to_temperature[0].weight[:, :-1], mean=0, std=0.01)
                nn.init.normal_(
                    self.knn_distance_to_temperature[0].weight[:, -1:], mean=0, std=0.1)

            # TODO we split the network here for different function, but may combine them in the future
            if self.knn_k_type == "trainable":

                self.knn_distance_to_k = nn.Sequential(
                    nn.Linear(args.max_k * 2 if self.label_count_as_feature else args.max_k,
                              args.knn_k_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=args.knn_k_net_dropout_rate),
                    # nn.Linear(args.knn_k_net_hid_size, args.max_k),
                    nn.Linear(args.knn_k_net_hid_size, args.max_k),
                    nn.Softmax(dim=-1))

                # nn.init.xavier_uniform_(self.knn_distance_to_k[0].weight, gain=0.01)
                # nn.init.xavier_uniform_(self.knn_distance_to_k[-2].weight, gain=0.01)
                # # TODO this maybe change or remove from here
                if self.label_count_as_feature:
                    nn.init.normal_(
                        self.knn_distance_to_k[0].weight[:, :args.max_k], mean=0, std=0.01)
                    nn.init.normal_(
                        self.knn_distance_to_k[0].weight[:, args.max_k:], mean=0, std=0.1)
                else:
                    nn.init.normal_(
                        self.knn_distance_to_k[0].weight, mean=0, std=0.01)

        self.append_adapter = args.decoder_append_adapter
        self.activate_adapter = args.activate_adapter
        self.only_knn_knowledge = False

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            # use_knn_store: bool = False
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            activate_adapter=self.activate_adapter
        )

        if self.use_knn_datastore or self.query_knn_datastore_during_training:
            last_hidden = x

        if not features_only:
            x = self.output_layer(x)

        # add by zhuwh
        if self.query_knn_datastore_during_training and self.compute_knn_loss:
            if self.args.use_all_gpus_to_search:
                knn_search_result = self.knn_datastore.retrieve(
                    last_hidden.float().cpu())
                knn_dists = knn_search_result['distance'][:, :, 1:]
                knn_index = knn_search_result['knn_index'][:, :, 1:]
                tgt_index = knn_search_result['tgt_index'][:, :, 1:]
                return x, extra, last_hidden, knn_dists, knn_index, tgt_index
            else:
                knn_search_result = self.knn_datastore.retrieve(
                    last_hidden.float())
                knn_dists = knn_search_result['distance'][:, :, 1:]
                knn_index = knn_search_result['knn_index'][:, :, 1:]
                tgt_index = knn_search_result['tgt_index'][:, :, 1:]
                return x, extra, last_hidden, knn_dists, knn_index, tgt_index

        if self.use_knn_datastore:
            # add by zhuwh
            if self.args.only_return_mt_knn_results:
                # get query results
                knn_search_result = self.knn_datastore.retrieve(
                    last_hidden)
                knn_dists = knn_search_result['distance']
                knn_index = knn_search_result['knn_index']
                tgt_index = knn_search_result['tgt_index']

                knn_temperature = self.knn_datastore.get_temperature()
                knn_lambda = self.knn_datastore.get_lambda()

                # get knn prob
                decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden,
                                                                      knn_temperature)
                knn_prob = decode_result['prob']

                # get mt prob
                logits = self.output_layer(last_hidden)
                mt_prob = F.softmax(logits, dim=-1)

                # get final prob
                final_prob = mt_prob*(1-knn_lambda) + knn_prob*knn_lambda

                return mt_prob, knn_prob, final_prob, last_hidden, knn_index, knn_dists, tgt_index

            # we should return the prob of knn search
            knn_search_result = self.knn_datastore.retrieve(last_hidden)
            # knn_probs = knn_search_result['prob']
            # [batch, seq len, k]  # we need do sort
            knn_dists = knn_search_result['distance']
            knn_index = knn_search_result['knn_index']
            tgt_index = knn_search_result['tgt_index']

            if self.label_count_as_feature:
                # TODO, we get the segment label count here, which is conflict with previous experiment
                label_counts = self.knn_datastore.get_label_count_segment(
                    tgt_index, relative=self.relative_label_count)
                network_inputs = torch.cat(
                    (knn_dists.detach(), label_counts.detach().float()), dim=-1)
            else:
                network_inputs = knn_dists.detach()

            if self.fp16:
                network_inputs = network_inputs.half()

            if self.knn_temperature_type == 'trainable':
                knn_temperature = None
            else:
                knn_temperature = self.knn_datastore.get_temperature()

            if self.knn_lambda_type == "trainable" and self.knn_k_type == 'trainable':
                net_outputs = self.retrieve_result_to_k_and_lambda(
                    network_inputs)

                k_prob = net_outputs  # [B, S, R_K]

                # we add this here only to test the effect of avg prob
                if self.avg_k:
                    k_prob = torch.zeros_like(
                        k_prob).fill_(1. / k_prob.size(-1))

                knn_lambda = 1. - k_prob[:, :, 0: 1]  # [B, S, 1]
                k_soft_prob = k_prob[:, :, 1:]
                decode_result = self.knn_datastore.calculate_select_knn_prob(knn_index, tgt_index, knn_dists,
                                                                             last_hidden,
                                                                             knn_temperature,
                                                                             k_soft_prob,
                                                                             is_test=not self.retrieve_result_to_k_and_lambda.training)

            else:
                if self.knn_lambda_type == 'trainable':
                    # self.knn_distances_to_lambda[2].p = 1.0

                    knn_lambda = self.knn_distances_to_lambda(network_inputs)

                else:
                    knn_lambda = self.knn_datastore.get_lambda()

                if self.knn_k_type == "trainable":
                    # we should generate k mask
                    k_prob = self.knn_distance_to_k(network_inputs)

                    if self.knn_distance_to_k.training:
                        # print(k_prob[0])
                        k_log_prob = torch.log(k_prob)
                        k_soft_one_hot = functional.gumbel_softmax(
                            k_log_prob, tau=0.1, hard=False, dim=-1)
                        # print(k_one_hot[0])

                    else:
                        # we get the one hot by argmax
                        _, max_idx = torch.max(k_prob, dim=-1)  # [B, S]
                        k_one_hot = torch.zeros_like(k_prob)
                        k_one_hot.scatter_(-1, max_idx.unsqueeze(-1), 1.)

                        knn_mask = torch.matmul(
                            k_one_hot, self.knn_datastore.mask_for_distance)

                if self.knn_k_type == "trainable" and self.knn_distance_to_k.training:
                    decode_result = self.knn_datastore.calculate_select_knn_prob(knn_index, tgt_index, knn_dists,
                                                                                 last_hidden,
                                                                                 knn_temperature,
                                                                                 k_soft_one_hot)

                elif self.knn_k_type == "trainable":
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden,
                                                                          knn_temperature, knn_mask)

                else:
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden,
                                                                          knn_temperature)

            knn_prob = decode_result['prob']

            if self.label_count_as_feature:
                return x, extra, knn_prob, knn_lambda, knn_dists, knn_index, label_counts
            else:
                return x, extra, knn_prob, knn_lambda, knn_dists, knn_index

        else:
            # original situation
            return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            activate_adapter: bool = False
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            activate_adapter=activate_adapter,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            activate_adapter: bool = False,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # add by zhuwh
        token_embedding = self.embed_tokens(prev_output_tokens)

        # embed tokens and positions
        x = self.embed_scale * token_embedding
        # x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                activate_adapter=activate_adapter
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(
                        name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output. we modify this method to return prob with
        knn result
        """

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(
                net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]

        # add by  , wo combine the knn prob and network prob here
        if self.use_knn_datastore:
            # x, extra, knn_probs, knn_lambda

            knn_probs = net_output[2]  # [batch, seq len, vocab size]
            knn_lambda = net_output[3]  # [batch, seq len, 1]
            # [batch, seq len, vocab size]
            network_probs = utils.softmax(
                logits, dim=-1, onnx_trace=self.onnx_trace)

            if self.knn_lambda_type == "fix":
                probs = network_probs * \
                    (1 - knn_lambda) + knn_probs * knn_lambda
            else:
                probs = network_probs * (1 - knn_lambda) + knn_probs

            if log_probs:
                return torch.log(probs)
            else:
                return probs

        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("transformer", "transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(
        args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(
        args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)


@register_model_architecture("transformer", "transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer", "transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer", "transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer", "transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer", "transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8192)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    transformer_wmt_en_de_big(args)


@register_model_architecture("transformer", "transformer_wmt19_de_en_with_datastore")
def transformer_wmt19_de_en_with_datastore(args):
    transformer_wmt19_de_en(args)