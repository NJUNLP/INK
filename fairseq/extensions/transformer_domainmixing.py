#

from .transformer_finetuning import base_architecture, TransformerModelForFinetuning

from fairseq.models import (
    register_model,
    register_model_architecture,
)

@register_model('transformer_domainmixing')
class DomainMixingTransformerModel(TransformerModelForFinetuning):
    pass


@register_model_architecture('transformer_domainmixing', 
                             'transformer_domainmixing')

def transformer_domainmixing(args):
    base_architecture(args)
