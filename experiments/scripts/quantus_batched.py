"""Batched forward passes for Quantus 0.6 PixelFlipping and Selectivity.

Quantus perturbs the input step by step and runs the model once per step on a batch of one image:
1344 VGG16 forwards per map for Pixel Flipping (112 features per step on 3x224x224), 784 for
Selectivity (8x8 patches). The GPU stays almost idle while each worker spends ~170 s per image.

install(chunk) replaces the two evaluate_batch methods with copies that build exactly the same
perturbed inputs, in the same order and with the same baseline at every step (the "mean" baseline
is recomputed from the already perturbed input, as in Quantus), and send them to the model `chunk`
steps at a time. The CPU builds the next chunk while the GPU runs the current one. Only the batching
of the forward pass changes; the scores match the original up to float rounding of the convolutions
(cuDNN picks other algorithms for other batch sizes). HM's code is not touched. CUDA only.
"""
import math

import numpy as np
import torch
from quantus.helpers import utils, warn
from quantus.metrics.faithfulness.pixel_flipping import PixelFlipping
from quantus.metrics.faithfulness.selectivity import Selectivity

CHUNK = 64      # as fast as 128 on an A6000, with half the GPU memory (~2.9 GiB peak)
IN_FLIGHT = 2   # chunks queued on the GPU at once; bounds the pinned host memory (~39 MB per chunk of 64)


class _Forward:
    """Runs model.predict of Quantus on chunks of steps, asynchronously, and collects the logits."""

    def __init__(self, model, x_batch_shape, y_batch):
        self.model, self.shape, self.y = model, x_batch_shape, y_batch
        self.pending, self.outs, self.events = [], [], []

    def add(self, x_perturbed, last):
        self.pending.append(x_perturbed)
        if len(self.pending) < CHUNK and not last:
            return
        if len(self.events) >= IN_FLIGHT:
            self.events.pop(0).synchronize()
        x = np.stack(self.pending).reshape(-1, *self.shape[1:])
        x = self.model.shape_input(x, x.shape, channel_first=True, batched=True)
        if self.model.model.training:
            raise AttributeError("Torch model needs to be in the evaluation mode.")
        with torch.no_grad():   # as PyTorchModel.predict, without its blocking .cpu()
            x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).pin_memory()
            self.outs.append(self.model.get_softmax_arg_model()(x.to(self.model.device, non_blocking=True)))
        self.events.append(torch.cuda.Event())
        self.events[-1].record()
        self.pending = []

    def result(self):
        # predictions of the target class, shape (n_steps, batch)
        b = self.shape[0]
        out = torch.cat(self.outs).cpu().numpy()
        return out.reshape(-1, b, out.shape[-1])[:, np.arange(b), self.y]


def _pixel_flipping_evaluate_batch(self, model, x_batch, y_batch, a_batch, **kwargs):
    # PixelFlipping.evaluate_batch of Quantus 0.6 with the model call moved out of the step loop
    if x_batch.shape != a_batch.shape:
        a_batch = np.broadcast_to(a_batch, x_batch.shape)
    batch_size = a_batch.shape[0]
    a_batch = a_batch.reshape(batch_size, -1)
    n_features = a_batch.shape[-1]
    a_indices = np.argsort(-a_batch, axis=1)

    n_perturbations = math.ceil(n_features / self.features_in_step)
    forward = _Forward(model, x_batch.shape, y_batch)
    x_perturbed = x_batch.copy()
    x_batch_shape = x_batch.shape
    for perturbation_step_index in range(n_perturbations):
        a_ix = a_indices[
            :,
            perturbation_step_index * self.features_in_step : (perturbation_step_index + 1) * self.features_in_step,
        ]
        x_perturbed = self.perturb_func(arr=x_perturbed.reshape(batch_size, -1), indices=a_ix)   # a new array
        x_perturbed = x_perturbed.reshape(*x_batch_shape)
        for x_element, x_perturbed_element in zip(x_batch, x_perturbed):
            warn.warn_perturbation_caused_no_change(x=x_element, x_perturbed=x_perturbed_element)
        forward.add(x_perturbed, last=perturbation_step_index == n_perturbations - 1)
    preds = forward.result()

    if self.return_auc_per_sample:
        return utils.calculate_auc(np.stack(preds, axis=1), batched=True).tolist()
    return np.stack(preds, axis=1).tolist()


def _selectivity_evaluate_batch(self, model, x_batch, y_batch, a_batch, **kwargs):
    # Selectivity.evaluate_batch of Quantus 0.6 with the model call moved out of the patch loop
    if x_batch.shape != a_batch.shape:
        a_batch = np.broadcast_to(a_batch, x_batch.shape)
    x_batch_shape = x_batch.shape
    if len(x_batch.shape) == 3:
        x_batch = x_batch[:, :, None]
        a_batch = a_batch[:, :, None]
    batch_size = a_batch.shape[0]
    x_perturbed = x_batch.copy()

    x_perturbed_h, x_perturbed_w = x_perturbed.shape[-2:]
    padding_h, padding_w = utils.get_padding_size(x_perturbed_h, self.patch_size), utils.get_padding_size(
        x_perturbed_w, self.patch_size
    )
    padding = ((0, 0), (0, 0), padding_h, padding_w)
    x_pad = utils._pad_array(x_batch, padding, mode="constant", padded_axes=np.arange(len(x_perturbed.shape)))
    a_pad = utils._pad_array(a_batch, padding, mode="constant", padded_axes=np.arange(len(x_perturbed.shape)))

    att_sums_list, patches_list = [], []
    for block_indices in utils.get_block_indices(x_pad, self.patch_size):
        a_sum = a_pad.reshape(batch_size, -1)[np.arange(batch_size)[:, None], block_indices].sum(axis=-1)
        att_sums_list.append(a_sum)
        patches_list.append(block_indices)
    att_sums = np.stack(att_sums_list, -1)
    patches = np.stack(patches_list, 1)
    order = np.argsort(-att_sums, -1)
    ordered_patches = patches[np.arange(batch_size)[:, None], order].transpose(1, 0, 2)

    forward = _Forward(model, x_batch_shape, y_batch)
    x_perturbed_pad = utils._pad_array(x_perturbed, padding, mode="edge", padded_axes=np.arange(len(x_perturbed.shape)))
    x_perturbed_pad_shape = x_perturbed_pad.shape
    for i, patch_slice in enumerate(ordered_patches):
        x_perturbed_pad = self.perturb_func(arr=x_perturbed_pad.reshape(batch_size, -1), indices=patch_slice)
        x_perturbed_pad = x_perturbed_pad.reshape(*x_perturbed_pad_shape)
        x_perturbed = x_perturbed_pad[
            :,
            :,
            padding_h[0] : x_perturbed_pad.shape[2] - padding_h[1],
            padding_w[0] : x_perturbed_pad.shape[3] - padding_w[1],
        ]
        for x_element, x_perturbed_element in zip(x_batch, x_perturbed):
            warn.warn_perturbation_caused_no_change(x=x_element, x_perturbed=x_perturbed_element)
        forward.add(x_perturbed.reshape(x_batch_shape), last=i == len(ordered_patches) - 1)
    return np.stack(list(forward.result()), 1).astype(np.float64)


def install(chunk=CHUNK):
    """Patch Quantus in this process; chunk <= 1 keeps the original one-step-at-a-time code."""
    global CHUNK
    if chunk <= 1:
        return
    assert torch.cuda.is_available(), "quantus_batched needs a CUDA device"
    CHUNK = int(chunk)
    PixelFlipping.evaluate_batch = _pixel_flipping_evaluate_batch
    Selectivity.evaluate_batch = _selectivity_evaluate_batch
