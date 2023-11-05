from mltoolkit.mlmo.utils.constants.checkpoint import MODEL_PARAMS, \
    OPTIMIZER_STATE
from mltoolkit.mlmo.interfaces import BaseIModel
from torch.nn import Module
from mltoolkit.mlmo.utils.helpers.loading_and_saving import load_embeddings
from torch.optim import Adam
from logging import getLogger
import torch as T
from torch.nn.utils import clip_grad_norm_
from mltoolkit.mlutils.tools.signature_scraper import repr_func
from mltoolkit.mlutils.helpers.general import select_matching_kwargs
from mltoolkit.mlmo.utils.helpers.pytorch.init import get_init_func
import os
from collections import OrderedDict
import utils
import numpy as np
from torch.cuda.amp import autocast, GradScaler

logger = getLogger(os.path.basename(__file__))


class ITorchModel(BaseIModel):
    """
    PyTorch model specific interface. Contains generic methods for training
    and evaluation.
    """

    def __init__(self, model, learning_rate, kl_anneal_portion,
                kl_const_portion, kl_const_coeff,num_total_iter,num_latent_scales,groups_per_scale,
                 weight_decay_norm_anneal,
                 weight_decay_norm_init,
                 weight_decay_norm, z_kl_ann,
                 device='cpu', optimizer=Adam,
                 grads_clip=None, **kwargs):
        """
        :param grads_clip: if float is passed will not allow gradients to exceed
                           a certain threshold. Allows to prevent gradients
                           explosion associated with RNNs.
        """
        if not isinstance(model, Module):
            raise ValueError("Please provide a valid PyTorch model.")
        super(ITorchModel, self).__init__(model, **kwargs)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.kl_anneal_portion = kl_anneal_portion
        self.kl_const_portion = kl_const_portion
        self.kl_const_coeff = kl_const_coeff
        self.num_total_iter = num_total_iter
        self.num_latent_scales = num_latent_scales
        self.groups_per_scale = groups_per_scale
        self.weight_decay_norm_anneal = weight_decay_norm_anneal
        self.weight_decay_norm_init = weight_decay_norm_init
        self.weight_decay_norm = weight_decay_norm
        self.z_kl_ann = z_kl_ann
        self.device = device
        self.model.to(device)
        logger.info("Moved the model to: '%s'" % device)
        self.scraper.scrape_obj_vals = False
        self.grads_clip = grads_clip
        self.model = model
        self.grad_scalar = GradScaler(2 ** 10)

    def train(self, batch, epoch, **kwrgs):
        """
        Performs a training step on a single batch. Returns the internal
        metrics in a dict, such as negative log-likelihood or KL divergence.
        """
        nelbo = utils.AvgrageMeter()
        self.model.train()  # setting the model to the train mode
        self.optimizer.zero_grad()  # zero the gradients before backward pass
        kwargs = select_matching_kwargs(self.model.forward, **batch.data)
        kwargs = _move_kwargs_to_device(device=self.device, **kwargs)
        _add_kwargs_to_dict(dct=kwargs, **kwrgs)

        alpha_i = utils.kl_balancer_coeff(num_scales=self.num_latent_scales,
                                          groups_per_scale=self.groups_per_scale, fun='square')

        rec_loss, kl_all, kl_diag, metrs = self.model(**kwargs)

        # dicte = self.model.state_dict()
        # print('dec_tower.3._ops.0.conv.4.weight', dicte['dec_tower.3._ops.0.conv.4.weight'][:10])


        # z_lambd = self.z_kl_ann(increment_indx=True)

        kl_coeff = utils.kl_coeff(epoch, self.kl_anneal_portion * self.num_total_iter,
                                  self.kl_const_portion * self.num_total_iter, self.kl_const_coeff)
        balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)


        loss = rec_loss + T.mean(balanced_kl)
        # loss = rec_loss

        norm_loss = self.model.spectral_norm_parallel()
        bn_loss = self.model.batchnorm_loss()


        # get spectral regularization coefficient (lambda)
        if self.weight_decay_norm_anneal:
            assert self.weight_decay_norm_init > 0 and self.weight_decay_norm > 0, 'init and final wdn should be positive.'
            wdn_coeff = (1. - kl_coeff) * np.log(self.weight_decay_norm_init) + kl_coeff * np.log(
                self.weight_decay_norm)
            wdn_coeff = np.exp(wdn_coeff)
        else:
            wdn_coeff = self.weight_decay_norm
        # loss += norm_loss * wdn_coeff
        loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        # loss = rec_loss
        # loss.backward()

        metrs['loss'] = loss.item()

        self.grad_scalar.scale(loss).backward()
        # utils.average_gradients(self.model.parameters(), True)
        # self.grad_scalar.update()
        nelbo.update(loss.data, 1)

        # clips the gradient
        if self.grads_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grads_clip)

        self.optimizer.step()

        return metrs

    def eval(self, batch, **kwrgs):
        """
        Performs computation of loss and internal metrics on a single batch.
        Same as training, but the model is not updated. Returns the internal
        metrics in a dict.
        """
        self.model.eval()  # setting the model to the test mode
        kwargs = select_matching_kwargs(self.model.forward, **batch.data)
        kwargs = _move_kwargs_to_device(device=self.device, **kwargs)
        _add_kwargs_to_dict(dct=kwargs, **kwrgs)
        with T.no_grad():
            rec_loss, kl_all, kl_diag, metrs = self.model(**kwargs)
        return metrs

    def save_state(self, file_path, excl_model_params=None):
        model_params = self.model.state_dict()
        optimizer_params = self.optimizer.state_dict()
        if excl_model_params is not None:
            for p in excl_model_params:
                del model_params[p]
        T.save({MODEL_PARAMS: model_params,
                OPTIMIZER_STATE: optimizer_params}, file_path)
        logger.info("Saved the model's and optimizer's state to: '%s'." %
                    file_path)

    def load_state(self, file_path, optimizer_state=False, strict=True):
        checkpoint = T.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint[MODEL_PARAMS], strict=strict)
        logger.info("Loaded the model's state from: '%s'." % file_path)
        if optimizer_state:
            self.optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE])
            logger.info("Loaded the optimizer's state from: '%s'." % file_path)

    def init_weights(self, multi_dim_init_func, single_dim_init_func):
        """Initializes weights using provided functions."""
        logger.info("Initializing multi-dim weights with:"
                    " %s." % repr_func(multi_dim_init_func))
        logger.info("Initializing single-dim weights with:"
                    " %s." % repr_func(single_dim_init_func))
        init = get_init_func(multi_dim_init_func, single_dim_init_func)
        self.model.apply(init)

    def init_embeddings(self, file_path, embds_layer_name, vocab):
        """Sets input and output embedding tensors with pre-trained ones."""
        embs = load_embeddings(file_path, vocab=vocab)
        embd_matr = T.tensor(embs).to(self.device)
        getattr(self.model, embds_layer_name).weight.data = embd_matr

    def __str__(self):
        return str(self.model) + "\n" + str(self.optimizer)


def _move_kwargs_to_device(device, **kwargs):
    for k in kwargs:
        kwargs[k] = kwargs[k].to(device)
    return kwargs


def _add_kwargs_to_dict(dct, **kwargs):
    """Adds new key-value pairs in-place to 'dct'."""
    for k, v in kwargs.items():
        if k in dct:
            raise ValueError("Key '%s' is already present in 'dct'.")
        dct[k] = v
