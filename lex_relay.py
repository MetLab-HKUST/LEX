""" One relay contains many sprints and is typically a complete simulation """

import numpy as np
import namelist_n_constants as nl
import setup_lex as setl
import one_sprint_functions as sp
import write_netcdf as write
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
write.save2nc(physState, sfcOthers, grids, 0, modelTime)
timeWrite = time.time() - wallTime

# do sprints
time1stSprint = 0.0
timeSprints = 0.0
for i in range(nl.relay_n):
    print("Sprint #%0.4i" % (i+1))
    wallTime = time.time()
    if nl.integrate_opt == 1:
        physState, sfcOthers = sp.ssprk3_sprint(physState, baseState, grids, modelOpt)
    elif nl.integrate_opt == 2:
        physState, sfcOthers = sp.leapfrog_sprint(physState, baseState, grids, modelOpt)

    if i > 0:
        timeSprints = time.time() - wallTime + timeSprints
    else:
        time1stSprint = time.time() - wallTime + time1stSprint

    wallTime = time.time()
    modelTime = np.round((i+1) * nl.sprint_n * nl.dt, decimals=6)
    write.save2nc(physState, sfcOthers, grids, (i+1), modelTime)
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
print("Other %4i Sprints: %10.6f" % (nl.relay_n-1, timeSprints))
print("Writing Data:       %10.6f" % timeWrite)
print("Total wall time:    %10.6f" % timeTotal)
print("-------------------------\n")

# end of the simulation
