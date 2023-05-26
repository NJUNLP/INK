import math

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

def weighted_label_smoothed_nll_loss(lprobs, target, epsilon, 
                                     ignore_index=None, reduce=True, 
                                     loss_weights=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if loss_weights is not None:
        nll_loss *= loss_weights
        smooth_loss *= loss_weights

    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('domain_weighting_lsce')
class DomainWeightingLSCECriterion(LabelSmoothedCrossEntropyCriterion):
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        parser.add_argument('--weighting-feature', default='domain', type=str,
                            help='the feature type for weighting lsce loss')
        # fmt: on

    def compute_loss(self, model, net_output, sample, reduce=True):
        return super().compute_loss(model, net_output, sample, reduce=reduce)
        #loss, nll_loss = super().compute_loss(model, net_output, sample, reduce=False)
        #loss = loss.reshape(sample['nsentences'], -1)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        loss_weights = sample[self.args.weighting_feature + '_weights'] # todo: compute weight from freq. 
        loss_weights = loss_weights.expand(-1, target.shape[-1])
        loss_weights = loss_weights.reshape(-1, 1)
        target = target.view(-1, 1)

        loss, nll_loss = weighted_label_smoothed_nll_loss(
            lprobs, target, self.eps, 
            ignore_index=self.padding_idx, reduce=reduce,
            loss_weights=loss_weights
        )
        return loss, nll_loss



