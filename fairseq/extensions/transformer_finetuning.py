#
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils

from fairseq.models.transformer import (
    TransformerModel,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    Embedding,
    Linear,
    TransformerEncoder,
    TransformerDecoder,
    base_architecture as _base_architecture,
    transformer_wmt_en_de_big,
)

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


def parse_embedding(embed_path):
    embed_dict = {}
    with open(embed_path, errors='ignore') as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]])

        # line = f_embed.readline()
        # while line:
        #     pieces = line.strip().split(" ")
        #     embed_dict[pieces[0]] = torch.Tensor([float(weight) for weight in pieces[1:]])
        #     try: # Ignore non-utf-8 tokens.
        #         line = f_embed.readline()
        #     except Exception as e:
        #         print(e, file=sys.stderr)
        #         continue
    return embed_dict

# def load_embedding(embed_dict, vocab, embedding):
#     for idx in range(len(vocab)):
#         token = vocab[idx]
#         if token in embed_dict:
#             embedding.weight.data[idx] = embed_dict[token]
#     return embedding


@register_model('transformer_finetuning')
class TransformerModelForFinetuning(TransformerModel):
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
        # * modified *
        parser.add_argument('--disable-training-embeddings', action='store_true',
                            help='if set, fix embedding layers'
                            ' (this requires to provide pretrained embeddings)')

        # fmt: on

    def __init__(self, encoder, decoder, extra_feature_dicts):
        super().__init__(encoder, decoder)
        self.extra_feature_dicts = extra_feature_dicts

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)

            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            if not path and args.disable_training_embeddings:
                raise ValueError(
                    'Do not set --disable_training_embeddings when pretrained embeddings are not provided.')

            if args.disable_training_embeddings:
                emb.weight.requires_grad = False
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError(
                    '--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path,
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        #self.extra_feature_dicts = task.extra_feature_dicts
        # return TransformerModelForFinetuning(encoder, decoder)
        return cls(encoder, decoder, task.extra_feature_dicts)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)


@register_model_architecture('transformer_finetuning', 'transformer_finetuning')
def base_architecture(args):
    _base_architecture(args)
    args.disable_training_embeddings = getattr(
        args, 'disable_training_embeddings', False)


@register_model_architecture("transformer_finetuning", "transformer_finetuning_wmt19_de_en")
def transformer_wmt19_de_en(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8192)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    transformer_wmt_en_de_big(args)
