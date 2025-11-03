""" Script to train the deep-learning subgrid-scale (DLS) model """

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".97"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import numpy as np
from flax.training import train_state
import optax
import orbax.checkpoint
import one_sprint_functions as sp
import setup_lex as setl
import namelist_n_constants as nl
import write_netcdf as write
import write_zarr
import dataset_generation as dgen
import pickle
import time


# setup initial conditions
print("Setting up I.C.")
startTime = time.time()
physState, baseState, grids, modelOpt = setl.setup_grid_n_ic(nl.ic_option)
timeSetup = time.time() - startTime

# do first-step integration
wallTime = time.time()
physState, sfcOthers = sp.first_step_integration_ssprk3(physState, baseState, grids, modelOpt)
time1stStep = time.time() - wallTime
print("First step integration done.")

# save base state and first one/two time slices
modelTime = 0.0
wallTime = time.time()
write.save2nc_base(baseState, grids)
if nl.output_format == 1:
    write.save2nc(physState, sfcOthers, grids, 0, modelTime)
elif nl.output_format == 2:
    write_zarr.save2zarr(physState, sfcOthers, 0, modelTime)
timeWrite = time.time() - wallTime

# load DL model and scaling parameters
with open("scaling_params_12_lookahead_steps_3_orig_steps.pkl", "rb") as f:
    scalingParams = pickle.load(f)
ckpt_dir = "/Workspace_I/shixm/LEX_DL/train_metlab3_no_theta_or_qv_limit/flax_checkpoints/lex_train_epoch0082"

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
# Build a target dict that matches what was saved
def create_train_state(rng, learning_rate):
    """ Create initial `TrainState` """
    dl_model = nl.dl_model
    params = dl_model.init(rng, jnp.ones([1, 40, 40, 40, 5]))['params']
    tx = optax.lamb(learning_rate)
    return train_state.TrainState.create(apply_fn=dl_model.apply, params=params, tx=tx)

RNG = jax.random.PRNGKey(2025)
RNG, initRNG = jax.random.split(RNG)
schedule = optax.constant_schedule(1.0e-5)
learningRate = schedule    # dummy rate when running the model
dlState = create_train_state(initRNG, learningRate)

# Build a target dict that matches what was saved
target = {
    'dl_state': dlState,
    'epoch': 0,
    'training_metrics': {
        'total_mse': 0.0,
        'grad_max': 0.0,
        'grad_min': 0.0,
        'grad_nan': False,
        'learning_rate': 0.0,
        },   # same structure as saved
    'testing_metrics':  {
        'total_mse': 0.0,
        },   # same structure as saved
}
# Build restore_args for that target structure
restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(target)
restored = orbax_checkpointer.restore(
    ckpt_dir,
    item=target,
    restore_args=restore_args,
)
dlState = restored['dl_state']
print("*** Saved DL state restored. ")


# do sprints
time1stSprint = 0.0
timeSprints = 0.0
for i in range(nl.relay_n):
    print("Sprint #%0.4i" % (i+1))
    wallTime = time.time()
    physState, sfcOthers = sp.ssprk3_sprint_dl(physState, baseState, grids, modelOpt, dlState.params, scalingParams)

    if i > 1:
        timeSprints = time.time() - wallTime + timeSprints
    else:
        time1stSprint = time.time() - wallTime + time1stSprint

    wallTime = time.time()
    modelTime = np.round((i+1) * nl.sprint_n * nl.dt, decimals=6)
    if nl.output_format == 1:
        write.save2nc(physState, sfcOthers, grids, (i+1), modelTime)
    elif nl.output_format == 2:
        write_zarr.save2zarr(physState, sfcOthers, (i+1), modelTime)
    timeWrite = time.time() - wallTime + timeWrite

endTime = time.time()
timeTotal = endTime - startTime

print("Completion of integration")
print("-------------------------")
print("Timing statistics")
print("***")
print("Setup:              %10.6f" % timeSetup)
print("1st Step:           %10.6f" % time1stStep)
print("1st & 2nd Sprints:  %10.6f" % time1stSprint)
print("Other %4i Sprints: %10.6f" % (nl.relay_n-1, timeSprints))
print("Writing Data:       %10.6f" % timeWrite)
print("Total wall time:    %10.6f" % timeTotal)
print("-------------------------\n")
