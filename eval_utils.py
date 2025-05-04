import torch
import logging
from torch import Tensor
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


def eval_model_loss(model, eval_dataloader, epoch_id, infer, args, step_id=None):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_sample = []
    pointwise_loss = []
    pointwise_sample = []
    tot_eq_stra = 0
    tot_stra = 0
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            loss_sample, n_sample, eq_stra, n_stra = model(
                validation=True,
                **batch
            ) # token_loss , label_size 
            if torch.isnan(loss_sample).sum().cpu().long().numpy() > 0:
                print(loss_sample)
                exit()
            tot_loss.append(loss_sample.sum().cpu().float().numpy())
            tot_sample.append(n_sample.sum().cpu().float().numpy())
            tot_eq_stra = tot_eq_stra + eq_stra
            tot_stra = tot_stra + n_stra
            if infer:
                pointwise_loss.extend(loss_sample.sum(dim=-1).cpu().tolist()) # 每一句 response 的 loss
                pointwise_sample.extend(n_sample.cpu().tolist()) # 每一句 response 的 实际length
    #exit()
    tot_loss = np.sum(tot_loss)
    tot_sample = np.sum(tot_sample)
    mean_loss = tot_loss / tot_sample
    mean_ppl = np.exp(mean_loss)
    stra_acc = tot_eq_stra / tot_stra
    print(f"\n Epoch {epoch_id}, Step {step_id}: Val loss {mean_loss} Val ppl {mean_ppl} Stra acc {stra_acc}")
    return mean_loss, mean_ppl, tot_sample, pointwise_loss, pointwise_sample,stra_acc