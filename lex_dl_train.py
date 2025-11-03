""" Single-GPU training for the DLS CNN (converted from Multi-GPU pmap) """

import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".97"
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import lax  
from functools import partial
import numpy as np
from flax import serialization, jax_utils  
from flax.training import train_state, orbax_utils, common_utils
import optax
import orbax.checkpoint
import time as Time
from one_sprint_functions import ssprk3_sprint_dl_train_vmap
import namelist_n_constants as nl
import setup_lex as setl
import dataset_generation as dgen


# --------------------------------------------------------------------------------
# Setup and data
# --------------------------------------------------------------------------------
# Get base state and grids
_, baseState, grids, modelOpt = setl.setup_grid_n_ic(nl.ic_option)

# Load dataset once
start = Time.time()
dataset, scalingParams = dgen.read_sim_data("Benchmark_Coarse_Grained.zarr", grids)
end = Time.time()
print('*** Master data loading time: %.2f' % (end - start))

# Devices (single-GPU training)
print(f"*** Single-device training. Detected devices: {[d.id for d in jax.devices()]}")

# --------------------------------------------------------------------------------
# (Removed) Utilities for sharding/replication (not needed in single-GPU)
# --------------------------------------------------------------------------------
# def shard_batch(...): ...
# def replicate_static(...): ...
# def unreplicate_static(...): ...

# --------------------------------------------------------------------------------
# Loss and helpers
# --------------------------------------------------------------------------------
def _loss_and_aux(params, ic, truth, scaling_params):
    theta_min, theta_max, u_min, u_max, v_min, v_max, w_min, w_max, qv_min, qv_max, th_la_min, th_la_max, qv_la_min, qv_la_max = scaling_params

    theta_batch, u_batch, v_batch, w_batch, pip_batch, qv_batch = ic

    # Adding ghost points & forward integration via your sprint function
    theta_stack, u_stack, v_stack, w_stack, pip_stack, qv_stack = ssprk3_sprint_dl_train_vmap(
        theta_batch, u_batch, v_batch, w_batch, pip_batch, qv_batch,
        baseState, grids, modelOpt, params, scaling_params
    )

    theta_true, u_true, v_true, w_true, pip_true, qv_true, del_th_true, del_qv_true = truth

    x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, _, _ = grids
    theta_p = theta_stack - theta_min    # strictly speaking, we should subtract base state
    del_th_stack = dgen.laplace_of_tensor_jnp(x3d, x3d4u, y3d, y3d4v, z3d, z3d4w, theta_p)
    qv_p = qv_stack - qv_min
    del_qv_stack = dgen.laplace_of_tensor_jnp(x3d, x3d4u, y3d, y3d4v, z3d, z3d4w, qv_p)

    # Loss
    loss = (jnp.sum(optax.l2_loss(theta_stack, theta_true) * (theta_max - theta_min) / (
                jnp.max(theta_max, axis=(2,3), keepdims=True) -
                jnp.min(theta_min, axis=(2,3), keepdims=True) + 0.01)**3) +
            jnp.sum(optax.l2_loss(u_stack, u_true) * (u_max - u_min) / (
                jnp.max(u_max, axis=(2,3), keepdims=True) -
                jnp.min(u_min, axis=(2,3), keepdims=True) + 0.01)**3) +
            jnp.sum(optax.l2_loss(v_stack, v_true) * (v_max - v_min) / (
                jnp.max(v_max, axis=(2,3), keepdims=True) -
                jnp.min(v_min, axis=(2,3), keepdims=True) + 0.01)**3) +
            jnp.sum(optax.l2_loss(w_stack, w_true) * (w_max - w_min) / (
                jnp.max(w_max, axis=(2,3), keepdims=True) -
                jnp.min(w_min, axis=(2,3), keepdims=True) + 0.01)**3) +
            jnp.sum(optax.l2_loss(qv_stack, qv_true) * (qv_max - qv_min) / (
                jnp.max(qv_max, axis=(2,3), keepdims=True) -
                jnp.min(qv_min, axis=(2,3), keepdims=True) + 1.0e-5)**3) +
            jnp.sum(optax.l2_loss(del_th_stack, del_th_true) * (th_la_max - th_la_min) / (
                jnp.max(th_la_max, axis=(2,3), keepdims=True)  -
                jnp.min(th_la_min, axis=(2,3), keepdims=True) + 1.0e-7)**3) +
            jnp.sum(optax.l2_loss(del_qv_stack, del_qv_true) * (qv_la_max - qv_la_min) / (
                jnp.max(qv_la_max, axis=(2,3), keepdims=True)  -
                jnp.min(qv_la_min, axis=(2,3), keepdims=True) + 1.0e-10)**3)
            )

    # Add a loss to penalize negative qv values.
    neg_qv = jnp.where(qv_stack<0.0, -qv_stack, 0.0)
    loss_neg_qv = jnp.sum( neg_qv * (qv_max - qv_min) / (
        jnp.max(qv_max, axis=(2,3), keepdims=True) -
        jnp.min(qv_min, axis=(2,3), keepdims=True) + 1.0e-5)**2)
    loss += loss_neg_qv
    
    return loss

def _grad_stats(grads):
    leaves = jax.tree_util.tree_leaves(grads)
    gmax = jnp.max(jnp.stack([jnp.max(x) for x in leaves]))
    gmin = jnp.min(jnp.stack([jnp.min(x) for x in leaves]))
    gnan = jnp.any(jnp.stack([jnp.any(jnp.isnan(x)) for x in leaves]))
    return gmax, gmin, gnan

# --------------------------------------------------------------------------------
# Single-device train/eval steps (jit)
# --------------------------------------------------------------------------------
@jax.jit
def train_step(dl_state, ic, truth, scaling_params):
    # Compute grads
    loss_grad_fn = jax.value_and_grad(_loss_and_aux)
    total_mse, grads = loss_grad_fn(dl_state.params, ic, truth, scaling_params)

    # Stats
    grad_max, grad_min, grad_nan = _grad_stats(grads)

    # Apply grads
    dl_state = dl_state.apply_gradients(grads=grads)
    lr = schedule(dl_state.step)

    metrics = {
        "total_mse": total_mse,
        "grad_max": grad_max,
        "grad_min": grad_min,
        "grad_nan": grad_nan,
        "learning_rate": lr
    }
    return dl_state, metrics


@jax.jit
def eval_step(dl_state, ic, truth, scaling_params):
    total_mse = _loss_and_aux(dl_state.params, ic, truth, scaling_params)
    return {"total_mse": total_mse}

# --------------------------------------------------------------------------------
# Epoch drivers (single-device, no sharding)
# --------------------------------------------------------------------------------
def train_epoch(dl_state, train_ic, train_true, epoch_size, batch_size, epoch, scaling_params):
    # Unpack epoch tensors
    theta_epoch, u_epoch, v_epoch, w_epoch, pip_epoch, qv_epoch = train_ic
    theta_true,  u_true,  v_true,  w_true,  pip_true,  qv_true, del_th_true, del_qv_true = train_true

    num_batches = int(epoch_size // batch_size)
    batch_metrics = []

    for i in range(num_batches):
        ic_batch = (
            theta_epoch[i*batch_size:(i+1)*batch_size, :, :, :],
            u_epoch[i*batch_size:(i+1)*batch_size,     :, :, :],
            v_epoch[i*batch_size:(i+1)*batch_size,     :, :, :],
            w_epoch[i*batch_size:(i+1)*batch_size,     :, :, :],
            pip_epoch[i*batch_size:(i+1)*batch_size,   :, :, :],
            qv_epoch[i*batch_size:(i+1)*batch_size,    :, :, :],
        )
        true_batch = (
            theta_true[i*batch_size:(i+1)*batch_size, :, :, :, :],
            u_true[i*batch_size:(i+1)*batch_size,     :, :, :, :],
            v_true[i*batch_size:(i+1)*batch_size,     :, :, :, :],
            w_true[i*batch_size:(i+1)*batch_size,     :, :, :, :],
            pip_true[i*batch_size:(i+1)*batch_size,   :, :, :, :],
            qv_true[i*batch_size:(i+1)*batch_size,    :, :, :, :],
            del_th_true[i*batch_size:(i+1)*batch_size,:, :, :, :],
            del_qv_true[i*batch_size:(i+1)*batch_size,:, :, :, :],
        )

        dl_state, metrics = train_step(dl_state, ic_batch, true_batch, scaling_params)
        # Move to host scalars for logging/aggregation
        batch_metrics.append(jax.device_get(metrics))

    training_epoch_metrics = {
        k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0]
    }

    print(">>> Training - epoch: %d" % epoch)
    print("    # of look-ahead steps: %d" % nl.sprint_n)
    print(">>> Training loss: %.4e" % training_epoch_metrics["total_mse"])
    print("    Learning rate: %.4e" % training_epoch_metrics["learning_rate"])
    print("    Gradient max: %.4e" % training_epoch_metrics["grad_max"])
    print("    Gradient min: %.4e" % training_epoch_metrics["grad_min"])
    print("    Any NaN gradient? %s" % bool(training_epoch_metrics["grad_nan"]))

    # 🔴 Stop training immediately if NaN gradient is found
    if bool(training_epoch_metrics["grad_nan"]):
        raise RuntimeError(
            f"!!! NaN detected in gradients at epoch {epoch}. Halting training !!!"
            )
        
    return dl_state, training_epoch_metrics


def eval_model(dl_state, test_ic, test_true, test_size, batch_size, epoch, scaling_params):
    theta_epoch, u_epoch, v_epoch, w_epoch, pip_epoch, qv_epoch = test_ic
    theta_true,  u_true,  v_true,  w_true,  pip_true,  qv_true, del_th_true, del_qv_true = test_true

    num_batches = int(test_size // batch_size)

    batch_metrics = []
    for i in range(num_batches):
        ic_batch = (
            theta_epoch[i*batch_size:(i+1)*batch_size, :, :, :],
            u_epoch[i*batch_size:(i+1)*batch_size,     :, :, :],
            v_epoch[i*batch_size:(i+1)*batch_size,     :, :, :],
            w_epoch[i*batch_size:(i+1)*batch_size,     :, :, :],
            pip_epoch[i*batch_size:(i+1)*batch_size,   :, :, :],
            qv_epoch[i*batch_size:(i+1)*batch_size,    :, :, :],
        )
        true_batch = (
            theta_true[i*batch_size:(i+1)*batch_size, :, :, :, :],
            u_true[i*batch_size:(i+1)*batch_size,     :, :, :, :],
            v_true[i*batch_size:(i+1)*batch_size,     :, :, :, :],
            w_true[i*batch_size:(i+1)*batch_size,     :, :, :, :],
            pip_true[i*batch_size:(i+1)*batch_size,   :, :, :, :],
            qv_true[i*batch_size:(i+1)*batch_size,    :, :, :, :],
            del_th_true[i*batch_size:(i+1)*batch_size,:, :, :, :],
            del_qv_true[i*batch_size:(i+1)*batch_size,:, :, :, :],
        )

        metrics = eval_step(dl_state, ic_batch, true_batch, scaling_params)
        batch_metrics.append(jax.device_get(metrics))

    testing_metrics = {
        k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0]
    }
    print('>>> Testing - epoch: %d' % epoch)
    print('>>> Testing loss: %.4e' % testing_metrics['total_mse'])
    return testing_metrics


# --------------------------------------------------------------------------------
# Orchestration (checkpointing)
# --------------------------------------------------------------------------------
RNG = jax.random.PRNGKey(2025)
RNG, initRNG = jax.random.split(RNG)

# Learning rate schedule & training config
schedule = nl.dl_schedule
learningRate = schedule
numEpochs = nl.dl_epochs
startEpoch = nl.dl_start_epoch
changeOptimizer = nl.dl_change_optimizer
batchSize = nl.dl_batch_size   # global batch size (no divisibility constraints now)

# --------------------------------------------------------------------------------
# Model & TrainState
# --------------------------------------------------------------------------------
def create_train_state(rng, learning_rate):
    """ Create initial `TrainState` """
    dl_model = nl.dl_model
    params = dl_model.init(rng, jnp.ones([1, 40, 40, 40, 5]))['params']
    tx = optax.lamb(learning_rate)
    return train_state.TrainState.create(apply_fn=dl_model.apply, params=params, tx=tx)

# Init state on host
dlState = create_train_state(initRNG, learningRate)

# Restart?
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
if startEpoch > 0:
    ckpt_dir = f"/Workspace_I/shixm/LEX_DL/train_metlab3/flax_checkpoints/lex_train_epoch{startEpoch:04d}"
    target = {
        'dl_state': dlState,
        'epoch': 0,
        'training_metrics': {
            'total_mse': 0.0,
            'grad_max': 0.0,
            'grad_min': 0.0,
            'grad_nan': False,
            'learning_rate': 0.0,
        },
        'testing_metrics':  {
            'total_mse': 0.0,
        },
    }
    restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(target)
    restored = orbax_checkpointer.restore(
        ckpt_dir,
        item=target,
        restore_args=restore_args,
    )
    dlState = restored['dl_state']
    if changeOptimizer:
        dl_model = nl.dl_model
        params = dlState.params
        tx = optax.lamb(learningRate)
        dlState = train_state.TrainState.create(apply_fn=dl_model.apply, params=params, tx=tx)

# --------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------
for Epoch in range(startEpoch, startEpoch + numEpochs):
    # (Re)generate epoch data (host arrays)
    start = Time.time()
    trainIC, trainTrue, testIC, testTrue, epochSize, testSize = dgen.generate_training_data(dataset, Epoch + 1)
    if Epoch == startEpoch:
        print(f"*** Training set size: {epochSize}; Testing set size: {testSize}")
    end = Time.time()
    print('*** Data generation time: %.2f' % (end - start))

    # Train
    start = Time.time()
    dlState, trainingMetrics = train_epoch(
        dlState, trainIC, trainTrue, epochSize, batchSize, Epoch + 1, scalingParams
    )
    end = Time.time()
    del trainIC, trainTrue
    print('*** Training time: %.2f' % (end - start))

    # Eval
    start = Time.time()
    testingMetrics = eval_model(
        dlState, testIC, testTrue, testSize, batchSize, Epoch + 1, scalingParams
    )
    end = Time.time()
    del testIC, testTrue
    print('*** Testing time: %.2f' % (end - start))

    # Save checkpoint
    start = Time.time()
    checkpoint = {
        'dl_state': dlState,
        'epoch': Epoch + 1,
        'training_metrics': trainingMetrics,
        'testing_metrics': testingMetrics
    }
    ckpt_dir = "/Workspace_I/shixm/LEX_DL/train_metlab3/flax_checkpoints/lex_train_epoch%0.4i" % (Epoch + 1)
    save_args = orbax_utils.save_args_from_target(checkpoint)
    orbax_checkpointer.save(ckpt_dir, checkpoint, save_args=save_args)
    end = Time.time()
    print('*** Saving checkpoint time: %.2f' % (end - start))
    print('==================================')
