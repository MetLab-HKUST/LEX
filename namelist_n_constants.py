""" Namelist variables and constants for LEX """
import dl_models as dlm
import optax

# Sovler setup
dt = 0.25            # the time interval (s) for one RK step integration
total_time = 1800    # total integration time (s)
save_time = 300      # time interval (s) between two data savings
solver_opt = 2       # 1: pseudo-incompressible equations;
                     # 2: fully compressible equations with expicit time-splitting
n_sound = 16         # number of acoustic steps in one RK step (for solver_opt=2)
sprint_n = 20        # the number of fused RK steps in one sprint

# Grid configuration
dx = 100.0   # x-direction grid spacing in meters
dy = 100.0   # y-direction grid spacing in meters
dz = 100.0   # z-direction grid spacing in meters
nx = 240     # number of grid cells in x-direction
ny = 240     # number of grid cells in y-direction
nz = 120     # number of grid cells in z-direction
ngx = 3      # number of ghost points on one side of the x-direction
ngy = 3      # number of ghost points on one side of the y-direction
ngz = 1      # number of ghost points on one side of the z-direction

# output file name
output_format = 1    # 1: netCDF4 (float32); 2: Zarr (float64)
file_name_format = "experiments/lex_out_%0.4i.nc"        # %0.4i prints the spring number
base_file_name = "experiments/lex_reference_state.nc"    # only netCDF4
save_num_levels = 1

# Initial condition choice
ic_option = 1
rand_opt = False

# Coriolis force
cor_opt = 0  # 0: no Coriolis; 1: with it; 2: with Coriolis+Large-scale pressure gradient
fCor = 1.0e-4   # Coriolis parameter

# Surface flux scheme
sfc_opt = False
atm_ocn_max_iter = 8    # maximum interation for the surface flux scheme
atm_ocn_min_wind = 0.25    # minimum wind allowed in the surface flux scheme
sfc_z_ref = 10.0    # reference height for wind, 10 m
sfc_z_t_ref = 2.0    # reference height for scalar, 2 m
Karman = 0.4    # von Karman constant

# Turbulence model
turb_opt = 0    # 1: Smagorinsky; 2: Anisotropic Smagorinsky

# Rayleigh damping
damp_opt = False
z_damping = 9600.0    # height above which Rayleigh damping is applied
rd_alpha = 1.0/120.0  # Inverse e-folding time for upper - level Rayleigh damping layer

# Diabatic heating
rad_opt = False      # turn on "radiation" or not

# pi' correction
pic_opt = False      # correction is needed if we need the actual density

# Physical constants
Rd = 287.04    # Gas constant of dry air
Rv = 461.5     # Gas constant of water vapor
Cp = 1004.64   # Specific heat of air at constant pressure
Cv = Cp-Rd     # Specific heat of air at constant volume
eps = Rd/Rv
Cpwv = 1810.0           # specific heat of water vapor
Cpvir = Cpwv / Cp - 1.0
reps = Rv/Rd
repsm1 = Rv/Rd - 1.0
lat_vap = 2.501e6       # latent heat of evaporation
g = 9.80616             # acceleration of gravity
p00 = 100000.0          # reference surface pressure

# Deep learning options
dl_model = dlm.AutoEncoder()
dl_schedule = optax.constant_schedule(1.0e-6)
dl_epochs = 12
dl_start_epoch = 82 
dl_change_optimizer = False    # if true, change optimizer in code manually in code
dl_batch_size = 4
dl_step_ratio = 3    # this is the number of benchmark data steps use by one step of the model being trained
