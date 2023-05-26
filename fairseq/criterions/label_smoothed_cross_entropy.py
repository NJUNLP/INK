# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy")
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model.decoder.compute_knn_loss = True
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)

        if model.args.query_knn_datastore_during_training:
            x, extra, last_hidden, knn_dists, knn_index, tgt_index = net_output
            target = sample['target']
            B, S, H = last_hidden.shape
            K = knn_index.shape[-1]
            knn_loss_type = model.args.knn_loss_type

            if knn_loss_type == 'kd_plus_cos':
                lprobs, _ = self.get_lprobs_and_target(model, net_output, sample)
                knn_index_flatten = knn_index.flatten().cpu().numpy()
                # cast knn_keys and last_hidden to float and back
                knn_keys = torch.tensor(
                    model.decoder.knn_datastore.keys[knn_index_flatten, :], 
                    requires_grad=False, 
                    device=x.device,
                    dtype=torch.float32)

                knn_keys = knn_keys.reshape(B, S, K, H)

                dist_func = torch.nn.CosineSimilarity(dim=-1)
                distance = dist_func(knn_keys, last_hidden.float().unsqueeze(-2))  # [B, S, K]
                distance_proxy = torch.exp(distance)
                # distance = torch.matmul(knn_keys, last_hidden.float().unsqueeze(-1)).squeeze(-1) # [B, S, K]
                
                knn_prob = F.softmax(distance_proxy[:,:,:8], dim=-1, dtype=torch.float32)
                _tgt_index = tgt_index[:,:,:8]
                place_holder = torch.zeros_like(lprobs)
                knn_soft_target = place_holder.scatter_add_(1, _tgt_index.view(B*S,-1), knn_prob.view(B*S, -1)) # avoid the error caused by in-place operate 
                kd_loss = F.kl_div(lprobs, knn_soft_target, reduction='none').sum(dim=-1)

                # compute knn loss
                knn_right_mask = (tgt_index.cuda() == target.unsqueeze(-1)).int()
                knn_loss = torch.sum(distance_proxy*knn_right_mask, dim=-1) / torch.sum(distance_proxy, dim=-1)  # [B, S]

                knn_all_wrong_mask = (torch.sum(knn_right_mask,dim=-1)==0)
                knn_loss.masked_fill_(knn_all_wrong_mask, 1)
                knn_loss = -torch.log(knn_loss)

                # compute final loss
                pad_mask = target.eq(self.padding_idx)  # [B, S]

                kd_loss.masked_fill_(pad_mask.view(B*S), 0.0)  # [B, S, K]
                kd_loss = torch.sum(kd_loss)

                knn_loss.masked_fill_(pad_mask, 0.0)  # [B, S, K]
                knn_loss = torch.sum(knn_loss)

                knn_loss_weight = model.args.knn_loss_weight
                kd_loss_weight = model.args.kd_loss_weight
                loss = loss + knn_loss_weight*knn_loss + kd_loss_weight*kd_loss

        if 'knn_loss' not in locals().keys():
            knn_loss = torch.tensor(0.0)
        if 'kd_loss' not in locals().keys():
            kd_loss = torch.tensor(0.0)

        model.decoder.compute_knn_loss = False
        sample_size = (
            sample["target"].size(
                0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "kd_loss": kd_loss.data,
            "knn_loss": knn_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )

        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        # add by
        knn_loss_sum = sum(log.get("knn_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "knn_loss", knn_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        kd_loss_sum = sum(log.get("kd_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "kd_loss", kd_loss_sum / ntokens / math.log(2), ntokens, round=3
        )


        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
