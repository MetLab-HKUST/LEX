""" Save model physical states to NetCDF files """

import netCDF4 as nc4
import numpy as np
import namelist_n_constants as nl


def save2nc(phys_state, grid_ic, sprint_i, model_time):
    """ Write data to a NetCDF file """
    filename = nl.fileNameFormat % sprint_i
    nc_file = nc4.Dataset(filename, mode="w", format="NETCDF4")
    nc_file.createDimension("x", nl.nx)
    nc_file.createDimension("y", nl.ny)
    nc_file.createDimension("z", nl.nz)
    nc_file.createDimension("x4u", (nl.nx + 1))
    nc_file.createDimension("y4v", (nl.ny + 1))
    nc_file.createDimension("z4w", (nl.nz + 1))
    nc_file.createDimension("time", None)

    x_nc = nc_file.createVariable("x", np.float32, ("x",))
    x_nc.long_name = "x (m)"
    y_nc = nc_file.createVariable("y", np.float32, ("y",))
    y_nc.long_name = "y (m)"
    z_nc = nc_file.createVariable("z", np.float32, ("z",))
    z_nc.long_name = "z (m)"
    x4u_nc = nc_file.createVariable("x4u", np.float32, ("x4u",))
    x4u_nc.long_name = "x (m) for u points"
    y4v_nc = nc_file.createVariable("y4v", np.float32, ("y4v",))
    y4v_nc.long_name = "y (m) for v points"
    z4w_nc = nc_file.createVariable("z4w", np.float32, ("z4w",))
    z4w_nc.long_name = "z (m) for w points"

    itime = nc_file.createVariable("time", np.float32, ("time",))
    itime.long_name = "time (s)"

    (rho0_theta0_prev, rho0_prev, theta0_prev,
     theta_prev, theta_now, pi0_prev, pip_prev,
     rhs, rhs_adv, rhs_cor, rhs_buoy, rhs_pres,
     qv_prev, qv_now, u_prev, u_now, v_prev, v_now, w_prev, w_now,
     info, pip_const, tau_x, tau_y, sen, evap, t_ref, q_ref, u10n) = phys_state
    (_, _, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, _, _) = grid_ic

    x_nc[:] = x3d[nl.ngx:-nl.ngx, 0, 0]
    y_nc[:] = y3d[0, nl.ngy:-nl.ngy, 0]
    z_nc[:] = z3d[0, 0, nl.ngz:-nl.ngz]
    x4u_nc[:] = x3d4u[nl.ngx:-nl.ngx, 0, 0]
    y4v_nc[:] = y3d4v[0, nl.ngy:-nl.ngy, 0]
    z4w_nc[:] = z3d4w[0, 0, nl.ngz:-nl.ngz]
    itime[:] = model_time

    x_size, y_size, z_size = np.shape(rho0_theta0_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    x4u_size, _, _ = np.shape(x3d4u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    _, y4v_size, _ = np.shape(y3d4v[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    _, _, z4w_size = np.shape(z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])

    rho0_theta0_1 = nc_file.createVariable("rho0_theta0_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    rho0_theta0_1.long_name = "rho0*theta0 for the current time"
    rho0_theta0_1[:, :, :, :] = np.reshape(rho0_theta0_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                           (1, x_size, y_size, z_size))

    theta0_1 = nc_file.createVariable("theta0_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    theta0_1.long_name = "theta0 for the current time"
    theta0_1[:, :, :, :] = np.reshape(theta0_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                      (1, x_size, y_size, z_size))

    theta_1 = nc_file.createVariable("theta_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    theta_1.long_name = "theta for the current time"
    theta_1[:, :, :, :] = np.reshape(theta_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                     (1, x_size, y_size, z_size))

    theta_2 = nc_file.createVariable("theta_next", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    theta_2.long_name = "theta for the next time step"
    theta_2[:, :, :, :] = np.reshape(theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                     (1, x_size, y_size, z_size))

    pi0_1 = nc_file.createVariable("pi0_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    pi0_1.long_name = "pi0 for the current time"
    pi0_1[:, :, :, :] = np.reshape(pi0_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                   (1, x_size, y_size, z_size))

    pip_1 = nc_file.createVariable("pip_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    pip_1.long_name = "pip for the current time"
    pip_1[:, :, :, :] = np.reshape(pip_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                   (1, x_size, y_size, z_size))

    rhs_1 = nc_file.createVariable("rhs_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    rhs_1.long_name = "rhs of pressure equation"
    rhs_1[:, :, :, :] = np.reshape(rhs, (1, x_size, y_size, z_size))

    rhs_adv_1 = nc_file.createVariable("rhs_adv_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    rhs_adv_1.long_name = "rhs adv of pressure equation"
    rhs_adv_1[:, :, :, :] = np.reshape(rhs_adv, (1, x_size, y_size, z_size))

    rhs_cor_1 = nc_file.createVariable("rhs_cor", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    rhs_cor_1.long_name = "rhs Coriolis of pressure equation"
    rhs_cor_1[:, :, :, :] = np.reshape(rhs_cor, (1, x_size, y_size, z_size))

    rhs_buoy_1 = nc_file.createVariable("rhs_buoy_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    rhs_buoy_1.long_name = "rhs buoyancy of pressure equation"
    rhs_buoy_1[:, :, :, :] = np.reshape(rhs_buoy, (1, x_size, y_size, z_size))

    rhs_pres_1 = nc_file.createVariable("rhs_pres_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    rhs_pres_1.long_name = "rhs buoyancy of pressure equation"
    rhs_pres_1[:, :, :, :] = np.reshape(rhs_pres, (1, x_size, y_size, z_size))
    
    qv_1 = nc_file.createVariable("qv_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    qv_1.long_name = "qv for the current time"
    qv_1[:, :, :, :] = np.reshape(qv_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x_size, y_size, z_size))

    qv_2 = nc_file.createVariable("qv_next", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36)
    qv_2.long_name = "qv for the next time step"
    qv_2[:, :, :, :] = np.reshape(qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x_size, y_size, z_size))

    u_1 = nc_file.createVariable("u_now", np.float32, ("time", "x4u", "y", "z"), fill_value=1.0e36)
    u_1.long_name = "u for the current time"
    u_1[:, :, :, :] = np.reshape(u_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x4u_size, y_size, z_size))

    u_2 = nc_file.createVariable("u_next", np.float32, ("time", "x4u", "y", "z"), fill_value=1.0e36)
    u_2.long_name = "u for the next time step"
    u_2[:, :, :, :] = np.reshape(u_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x4u_size, y_size, z_size))

    v_1 = nc_file.createVariable("v_now", np.float32, ("time", "x", "y4v", "z"), fill_value=1.0e36)
    v_1.long_name = "v for the current time"
    v_1[:, :, :, :] = np.reshape(v_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x_size, y4v_size, z_size))

    v_2 = nc_file.createVariable("v_next", np.float32, ("time", "x", "y4v", "z"), fill_value=1.0e36)
    v_2.long_name = "v for the next time step"
    v_2[:, :, :, :] = np.reshape(v_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x_size, y4v_size, z_size))

    w_1 = nc_file.createVariable("w_now", np.float32, ("time", "x", "y", "z4w"), fill_value=1.0e36)
    w_1.long_name = "w for the current time"
    w_1[:, :, :, :] = np.reshape(w_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x_size, y_size, z4w_size))

    w_2 = nc_file.createVariable("w_next", np.float32, ("time", "x", "y", "z4w"), fill_value=1.0e36)
    w_2.long_name = "w for the next time step"
    w_2[:, :, :, :] = np.reshape(w_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x_size, y_size, z4w_size))

    info_1 = nc_file.createVariable("info_now", np.float32, "time", fill_value=1.0e36)
    info_1.long_name = "exit code of the Poisson-like equation solver"
    info_1[:] = info

    pc_1 = nc_file.createVariable("pip_const_now", np.float32, "time", fill_value=1.0e36)
    pc_1.long_name = "pi' correction constant for the current time"
    pc_1[:] = pip_const

    tau_x_1 = nc_file.createVariable("tau_x_now", np.float32, ("time", "x", "y"), fill_value=1.0e36)
    tau_x_1.long_name = "tau_x for the current time"
    tau_x_1[:, :, :] = np.reshape(tau_x, (1, x_size, y_size))

    tau_y_1 = nc_file.createVariable("tau_y_now", np.float32, ("time", "x", "y"), fill_value=1.0e36)
    tau_y_1.long_name = "tau_y for the current time"
    tau_y_1[:, :, :] = np.reshape(tau_y, (1, x_size, y_size))

    sen_1 = nc_file.createVariable("sen_now", np.float32, ("time", "x", "y"), fill_value=1.0e36)
    sen_1.long_name = "sensible heat for the current time"
    sen_1[:, :, :] = np.reshape(sen, (1, x_size, y_size))

    evap_1 = nc_file.createVariable("evap_now", np.float32, ("time", "x", "y"), fill_value=1.0e36)
    evap_1.long_name = "evaporation for the current time"
    evap_1[:, :, :] = np.reshape(evap, (1, x_size, y_size))

    t_ref_1 = nc_file.createVariable("T_ref_now", np.float32, ("time", "x", "y"), fill_value=1.0e36)
    t_ref_1.long_name = "temperature at reference height (2m) for the current time"
    t_ref_1[:, :, :] = np.reshape(t_ref, (1, x_size, y_size))

    q_ref_1 = nc_file.createVariable("q_ref_now", np.float32, ("time", "x", "y"), fill_value=1.0e36)
    q_ref_1.long_name = "mixing ratio at reference height (2m) for the current time"
    q_ref_1[:, :, :] = np.reshape(q_ref, (1, x_size, y_size))

    u10_1 = nc_file.createVariable("u10_now", np.float32, ("time", "x", "y"), fill_value=1.0e36)
    u10_1.long_name = "wind speed at reference height (10m) for the current time"
    u10_1[:, :, :] = np.reshape(u10n, (1, x_size, y_size))

    nc_file.close()

    return filename
