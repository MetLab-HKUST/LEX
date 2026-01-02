""" One relay contains many sprints and is typically a complete simulation """

import jax
jax.config.update("jax_enable_x64", True)  # use double precision if necessary
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import namelist_n_constants as nl
import setup_lex as setl
import one_sprint_functions as sp
import write_netcdf as write
import write_zarr
import time


# setup initial conditions
print("Setting up I.C.")
startTime = time.time()
physState, baseState, grids, modelOpt = setl.setup_grid_n_ic(nl.ic_option)
timeSetup = time.time() - startTime

# get first-step flux
wallTime = time.time()
sfcOthers = sp.get_first_step_flux(physState, baseState, grids, modelOpt)
time1stStep = time.time() - wallTime
print("First step flux done.")

# save base state and first one/two time slices
modelTime = 0.0
wallTime = time.time()
write.save2nc_base(baseState, grids)
if nl.output_format == 1:
    write.save2nc(physState, sfcOthers, grids, 0, modelTime)
elif nl.output_format == 2:
    write_zarr.save2zarr(physState, sfcOthers, 0, modelTime)
timeWrite = time.time() - wallTime

# do sprints
relay_n = round(nl.total_time/nl.dt/nl.sprint_n)   # total number of sprints
save_n = round(nl.save_time/nl.dt/nl.sprint_n)     # number of relays between data saving windows
time1stSprint = 0.0
timeSprints = 0.0

for i in range(relay_n):
    print("Sprint #%0.4i" % (i+1))
    wallTime = time.time()
    physState, sfcOthers = sp.ssprk3_sprint(physState, baseState, grids, modelOpt)
    if i > 0:
        timeSprints = time.time() - wallTime + timeSprints
    else:
        time1stSprint = time.time() - wallTime + time1stSprint

    if ((i+1)%save_n == 0):
        wallTime = time.time()
        modelTime = np.round((i+1) * nl.sprint_n * nl.dt, decimals=6)
        if nl.output_format == 1:
            write.save2nc(physState, sfcOthers, grids, (i+1)/save_n, modelTime)
        elif nl.output_format == 2:
            write_zarr.save2zarr(physState, sfcOthers, (i+1)/save_n, modelTime)
        timeWrite = time.time() - wallTime + timeWrite

endTime = time.time()
timeTotal = endTime - startTime

print("Completion of integration")
print("-------------------------")
print("Timing statistics")
print("***")
print("Setup:              %10.6f" % timeSetup)
print("1st Step:           %10.6f" % time1stStep)
print("1st Sprint:         %10.6f" % time1stSprint)
print("Other %4i Sprints: %10.6f" % (relay_n-1, timeSprints))
print("Writing Data:       %10.6f" % timeWrite)
print("Total wall time:    %10.6f" % timeTotal)
print("-------------------------\n")

# end of the simulation
