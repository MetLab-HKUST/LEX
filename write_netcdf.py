""" Save model physical states to NetCDF files """

import netCDF4 as nc4
import numpy as np
import namelist_n_constants as nl


def save2nc(phys_state, sfc_others, grids, sprint_i, model_time):
    """ Write data to a NetCDF file """
    filename = nl.file_name_format % sprint_i
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

    if nl.integrate_opt == 1:
        (theta_now, u_now, v_now, w_now, pip_now, qv_now) = phys_state    # pip_now is actually the previous step pi'
    elif nl.integrate_opt == 2:
        (theta_now, theta_next, pip_now, qv_now, qv_next, u_now, u_next, v_now, v_next, w_now, w_next) = phys_state

    (info, pip_const, tau_x, tau_y, sen, evap, t_ref, q_ref, u10n) = sfc_others
    x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, tauh, tauf = grids

    x_nc[:] = x3d[nl.ngx:-nl.ngx, 0, 0]
    y_nc[:] = y3d[0, nl.ngy:-nl.ngy, 0]
    z_nc[:] = z3d[0, 0, nl.ngz:-nl.ngz]
    x4u_nc[:] = x3d4u[nl.ngx:-nl.ngx, 0, 0]
    y4v_nc[:] = y3d4v[0, nl.ngy:-nl.ngy, 0]
    z4w_nc[:] = z3d4w[0, 0, nl.ngz:-nl.ngz]
    itime[:] = model_time

    x_size, y_size, z_size = nl.nx, nl.ny, nl.nz
    x4u_size = nl.nx + 1
    y4v_size = nl.ny + 1
    z4w_size = nl.nz + 1

    theta_1 = nc_file.createVariable("theta_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    theta_1.long_name = "theta for the current time"
    theta_1[:, :, :, :] = np.reshape(theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                     (1, x_size, y_size, z_size))

    pip_1 = nc_file.createVariable("pip_now", np.float64, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    pip_1.long_name = "pip for the current time"
    pip_1[:, :, :, :] = np.reshape(pip_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                   (1, x_size, y_size, z_size))

    qv_1 = nc_file.createVariable("qv_now", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    qv_1.long_name = "qv for the current time"
    qv_1[:, :, :, :] = np.reshape(qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x_size, y_size, z_size))

    u_1 = nc_file.createVariable("u_now", np.float32, ("time", "x4u", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    u_1.long_name = "u for the current time"
    u_1[:, :, :, :] = np.reshape(u_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x4u_size, y_size, z_size))

    v_1 = nc_file.createVariable("v_now", np.float32, ("time", "x", "y4v", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    v_1.long_name = "v for the current time"
    v_1[:, :, :, :] = np.reshape(v_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x_size, y4v_size, z_size))

    w_1 = nc_file.createVariable("w_now", np.float32, ("time", "x", "y", "z4w"), fill_value=1.0e36, zlib=True, complevel=2)
    w_1.long_name = "w for the current time"
    w_1[:, :, :, :] = np.reshape(w_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], (1, x_size, y_size, z4w_size))

    info_1 = nc_file.createVariable("info_now", np.float32, "time", fill_value=1.0e36, zlib=True, complevel=2)
    info_1.long_name = "exit code of the Poisson equation solver"
    info_1[:] = info

    if nl.save_num_levels == 2 and nl.integrate_opt != 1:
        theta_2 = nc_file.createVariable("theta_next", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
        theta_2.long_name = "theta for the next time step"
        theta_2[:, :, :, :] = np.reshape(theta_next[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                         (1, x_size, y_size, z_size))

        qv_2 = nc_file.createVariable("qv_next", np.float32, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
        qv_2.long_name = "qv for the next time step"
        qv_2[:, :, :, :] = np.reshape(qv_next[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                      (1, x_size, y_size, z_size))

        u_2 = nc_file.createVariable("u_next", np.float32, ("time", "x4u", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
        u_2.long_name = "u for the next time step"
        u_2[:, :, :, :] = np.reshape(u_next[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                     (1, x4u_size, y_size, z_size))

        v_2 = nc_file.createVariable("v_next", np.float32, ("time", "x", "y4v", "z"), fill_value=1.0e36, zlib=True, complevel=2)
        v_2.long_name = "v for the next time step"
        v_2[:, :, :, :] = np.reshape(v_next[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                     (1, x_size, y4v_size, z_size))

        w_2 = nc_file.createVariable("w_next", np.float32, ("time", "x", "y", "z4w"), fill_value=1.0e36, zlib=True, complevel=2)
        w_2.long_name = "w for the next time step"
        w_2[:, :, :, :] = np.reshape(w_next[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                     (1, x_size, y_size, z4w_size))

    if nl.pic_opt:
        pc_1 = nc_file.createVariable("pip_const_now", np.float64, "time", fill_value=1.0e36, zlib=True, complevel=2)
        pc_1.long_name = "pi' correction constant for the current time"
        pc_1[:] = pip_const

    if nl.sfc_opt:
        tau_x_1 = nc_file.createVariable("tau_x_now", np.float32, ("time", "x", "y"), fill_value=1.0e36, zlib=True, complevel=2)
        tau_x_1.long_name = "tau_x for the current time"
        tau_x_1[:, :, :] = np.reshape(tau_x, (1, x_size, y_size))

        tau_y_1 = nc_file.createVariable("tau_y_now", np.float32, ("time", "x", "y"), fill_value=1.0e36, zlib=True, complevel=2)
        tau_y_1.long_name = "tau_y for the current time"
        tau_y_1[:, :, :] = np.reshape(tau_y, (1, x_size, y_size))

        sen_1 = nc_file.createVariable("sen_now", np.float32, ("time", "x", "y"), fill_value=1.0e36, zlib=True, complevel=2)
        sen_1.long_name = "sensible heat for the current time"
        sen_1[:, :, :] = np.reshape(sen, (1, x_size, y_size))

        evap_1 = nc_file.createVariable("evap_now", np.float32, ("time", "x", "y"), fill_value=1.0e36, zlib=True, complevel=2)
        evap_1.long_name = "evaporation for the current time"
        evap_1[:, :, :] = np.reshape(evap, (1, x_size, y_size))

        t_ref_1 = nc_file.createVariable("T_ref_now", np.float32, ("time", "x", "y"), fill_value=1.0e36, zlib=True, complevel=2)
        t_ref_1.long_name = "temperature at reference height (2m) for the current time"
        t_ref_1[:, :, :] = np.reshape(t_ref, (1, x_size, y_size))

        q_ref_1 = nc_file.createVariable("q_ref_now", np.float32, ("time", "x", "y"), fill_value=1.0e36, zlib=True, complevel=2)
        q_ref_1.long_name = "mixing ratio at reference height (2m) for the current time"
        q_ref_1[:, :, :] = np.reshape(q_ref, (1, x_size, y_size))

        u10_1 = nc_file.createVariable("u10_now", np.float32, ("time", "x", "y"), fill_value=1.0e36, zlib=True, complevel=2)
        u10_1.long_name = "wind speed at reference height (10m) for the current time"
        u10_1[:, :, :] = np.reshape(u10n, (1, x_size, y_size))

    nc_file.close()

    return filename


def save2nc_base(base_state, grids):
    """ Write data to a NetCDF file """
    filename = nl.base_file_name
    nc_file = nc4.Dataset(filename, mode="w", format="NETCDF4")
    nc_file.createDimension("x", nl.nx)
    nc_file.createDimension("y", nl.ny)
    nc_file.createDimension("z", nl.nz)
    nc_file.createDimension("time", None)

    x_nc = nc_file.createVariable("x", np.float64, ("x",))
    x_nc.long_name = "x (m)"
    y_nc = nc_file.createVariable("y", np.float64, ("y",))
    y_nc.long_name = "y (m)"
    z_nc = nc_file.createVariable("z", np.float64, ("z",))
    z_nc.long_name = "z (m)"

    itime = nc_file.createVariable("time", np.float64, ("time",))
    itime.long_name = "time (s)"

    rho0_theta0, rho0, theta0, pi0, qv0, surface_t = base_state
    x3d, y3d, z3d, _, _, _, _, _ = grids

    x_nc[:] = x3d[nl.ngx:-nl.ngx, 0, 0]
    y_nc[:] = y3d[0, nl.ngy:-nl.ngy, 0]
    z_nc[:] = z3d[0, 0, nl.ngz:-nl.ngz]
    itime[:] = 0.0

    x_size, y_size, z_size = nl.nx, nl.ny, nl.nz

    rho0_theta0_nc = nc_file.createVariable("rho0_theta0", np.float64, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    rho0_theta0_nc.long_name = "rho0*theta0"
    rho0_theta0_nc[:, :, :, :] = np.reshape(rho0_theta0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                            (1, x_size, y_size, z_size))

    theta0_nc = nc_file.createVariable("theta0", np.float64, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    theta0_nc.long_name = "theta0"
    theta0_nc[:, :, :, :] = np.reshape(theta0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                       (1, x_size, y_size, z_size))

    rho0_nc = nc_file.createVariable("rho0", np.float64, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    rho0_nc.long_name = "rho0"
    rho0_nc[:, :, :, :] = np.reshape(rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                     (1, x_size, y_size, z_size))

    qv0_nc = nc_file.createVariable("qv0", np.float64, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    qv0_nc.long_name = "qv0"
    qv0_nc[:, :, :, :] = np.reshape(qv0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                    (1, x_size, y_size, z_size))

    pi0_nc = nc_file.createVariable("pi0", np.float64, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    pi0_nc.long_name = "pi0"
    pi0_nc[:, :, :, :] = np.reshape(pi0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                    (1, x_size, y_size, z_size))

    sst = nc_file.createVariable("T_sfc", np.float64, ("time", "x", "y"), fill_value=1.0e36, zlib=True, complevel=2)
    sst.long_name = "surface temperature"
    sst[:, :, :] = np.reshape(surface_t, (1, x_size, y_size))

    nc_file.close()

    return filename
