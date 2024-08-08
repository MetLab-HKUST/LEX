""" Functions for one step of the LEX model integration """

import jax
import jax.numpy as jnp
import advection as adv
import namelist_n_constants as nl
import radiation as rad
import pressure_equations as pres_eqn
import pressure_gradient_coriolis as pres_grad
import boundary_conditions as bc


def update_rho0_theta0_euler(rho0_theta0_now, rho0_now, u_now, v_now, w_now, heating_now, x3d4u, y3d4v, z3d4w):
    """ Obtain rho0*theta0, rho0, theta0, pi0, and rho0*heating for the next step """
    d_rho0theta0_dt_now, rho0_theta0_heating_now = compute_rho0_theta0_tendency(rho0_theta0_now, rho0_now, u_now, v_now,
                                                                                w_now, heating_now, x3d4u, y3d4v, z3d4w)
    rho0_theta0_next = rho0_theta0_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + d_rho0theta0_dt_now * nl.dt
    pi0_next = (rho0_theta0_next * nl.Rd / nl.p00) ** (nl.Rd / nl.Cv)
    pi0_next8w = pres_eqn.interpolate_scalar2w(pi0_next)
    d_pi0_dz = (pi0_next8w[:, :, 1:] - pi0_next8w[:, :, 0:-1]) / (
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] -
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz + 1)])
    theta0_next = -nl.g / nl.Cp / d_pi0_dz
    rho0_next = rho0_theta0_next / theta0_next
    # update ghost points
    rho0_theta0_next3 = padding3_array(rho0_theta0_next)
    rho0_next3 = padding3_array(rho0_next)
    theta0_next3 = padding3_array(theta0_next)
    pi0_next3 = padding3_array(pi0_next)
    return rho0_theta0_next3, rho0_next3, theta0_next3, pi0_next3, rho0_theta0_heating_now, d_rho0theta0_dt_now


def update_rho0_theta0_leapfrog(rho0_theta0_prev, rho0_theta0_now, rho0_now, u_now, v_now, w_now, heating_now,
                                x3d4u, y3d4v, z3d4w):
    """ Obtain rho0*theta0, rho0, theta0, pi0, and rho0*heating for the next step """
    d_rho0theta0_dt_now, rho0_theta0_heating_now = compute_rho0_theta0_tendency(rho0_theta0_now, rho0_now, u_now, v_now,
                                                                                w_now, heating_now, x3d4u, y3d4v, z3d4w)
    rho0_theta0_next = rho0_theta0_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy,
                                        nl.ngz:-nl.ngz] + d_rho0theta0_dt_now * 2.0 * nl.dt
    pi0_next = (rho0_theta0_next * nl.Rd / nl.p00) ** (nl.Rd / nl.Cv)
    pi0_next8w = pres_eqn.interpolate_scalar2w(pi0_next)
    d_pi0_dz = (pi0_next8w[:, :, 1:] - pi0_next8w[:, :, 0:-1]) / (
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] -
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz + 1)])
    theta0_next = -nl.g / nl.Cp / d_pi0_dz
    rho0_next = rho0_theta0_next / theta0_next
    # update ghost points
    rho0_theta0_next3 = padding3_array(rho0_theta0_next)
    rho0_next3 = padding3_array(rho0_next)
    theta0_next3 = padding3_array(theta0_next)
    pi0_next3 = padding3_array(pi0_next)
    return rho0_theta0_next3, rho0_next3, theta0_next3, pi0_next3, rho0_theta0_heating_now, d_rho0theta0_dt_now


def update_theta_euler(rho0_now, theta_now, u_now, v_now, w_now, flow_divergence, sfc_flux, heating, x3d4u, y3d4v,
                       z3d4w):
    """ Update theta to get the next-step values """
    d_theta_dt = compute_theta_tendency(rho0_now, theta_now, u_now, v_now, w_now, flow_divergence, sfc_flux, heating,
                                        x3d4u, y3d4v, z3d4w)
    theta_next = theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + d_theta_dt * nl.dt
    theta_next3 = padding3_array(theta_next)
    return theta_next3


def update_theta_leapfrog(theta_prev, rho0_now, theta_now, u_now, v_now, w_now, flow_divergence, sfc_flux, heating,
                          x3d4u, y3d4v, z3d4w):
    """ Update theta to get the next-step values """
    d_theta_dt = compute_theta_tendency(rho0_now, theta_now, u_now, v_now, w_now, flow_divergence, sfc_flux, heating,
                                        x3d4u, y3d4v, z3d4w)
    theta_next = theta_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + d_theta_dt * 2.0 * nl.dt
    theta_next3 = padding3_array(theta_next)
    return theta_next3


def update_qv_euler(rho0_now, qv_now, u_now, v_now, w_now, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w):
    """ Update theta to get the next-step values """
    d_qv_dt = compute_qv_tendency(rho0_now, qv_now, u_now, v_now, w_now, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w)
    qv_next = qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + d_qv_dt * nl.dt
    qv_next3 = padding3_array(qv_next)
    return qv_next3


def update_qv_leapfrog(qv_prev, rho0_now, qv_now, u_now, v_now, w_now, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w):
    """ Update theta to get the next-step values """
    d_qv_dt = compute_qv_tendency(rho0_now, qv_now, u_now, v_now, w_now, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w)
    qv_next = qv_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + d_qv_dt * 2.0 * nl.dt
    qv_next3 = padding3_array(qv_next)
    return qv_next3


def solve_pres_eqn(pip_prev, rho0_theta0, pi0, rtt, u, v, w, adv4u, adv4v, adv4w, fu, fv, buoyancy,
                   rho0_theta0_heating1, rho0_theta0_heating2, rho0_theta0_tend1, rho0_theta0_tend2,
                   x3d, x3d4u, y3d, y3d4v, z3d4w):
    """ Solve the Poisson-like equation for pressure perturbations, pi\' (pip) """
    rhs = pres_eqn.rhs_of_pressure_equation(rho0_theta0, pi0, rtt, u, v, w, adv4u, adv4v, adv4w, fu, fv, buoyancy,
                                            rho0_theta0_heating1, rho0_theta0_heating2, rho0_theta0_tend1,
                                            rho0_theta0_tend2,
                                            x3d, x3d4u, y3d, y3d4v, z3d4w)

    tol = 1.0e-6  # the tolerance level needs to be tested and tuned.
    # using previous step pi\' as the first guess x0
    pip = jax.scipy.sparse.linalg.gmres(pres_eqn.laplace_of_pressure, rhs, x0=pip_prev, tol=tol, maxiter=100,
                                        solve_method='incremental')
    return pip


def correct_pip_constant(rho0_prev, theta0_prev, pi0_prev, rho0_next, theta0_next, pi0_next, pi0_now, pip_now,
                         x3d4u, y3d4v, z3d4w):
    """ Add a constant to the solution to the Poisson-like equation for pressure perturbations """
    t0_prev = pi0_prev * theta0_prev
    t0_next = pi0_next * theta0_next
    phi = 1.0 / pi0_now * (rho0_next * t0_next - rho0_prev * t0_prev)
    numerator = -space_integration(pip_now * phi, x3d4u, y3d4v, z3d4w)
    denominator = space_integration(phi, x3d4u, y3d4v, z3d4w)
    return numerator/denominator


def update_momentum_eqn_euler(u_now, v_now, w_now, pi0_now, pip_now, theta_now, adv4u, adv4v, adv4w, fu, fv, b8w, x3d,
                              y3d, z3d):
    """ Update momentum to get the next-step values """
    pres_grad4u, pres_grad4v, pres_grad4w = pres_grad.pressure_gradient_force(pi0_now, pip_now, theta_now, x3d, y3d,
                                                                              z3d)

    u_next = u_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + (adv4u + pres_grad4u + fv) * nl.dt
    v_next = v_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + (adv4v + pres_grad4v - fu) * nl.dt
    w_next = w_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + (adv4w + pres_grad4w + b8w) * nl.dt
    u_next3 = padding3_array(u_next)
    v_next3 = padding3_array(v_next)
    w_next3 = padding3_array(w_next)
    w_next3 = bc.set_w_bc(w_next3)
    return u_next3, v_next3, w_next3


def update_momentum_eqn_leapfrog(u_prev, v_prev, w_prev, pi0_now, pip_now, theta_now, adv4u, adv4v, adv4w, fu, fv, b8w,
                                 x3d, y3d, z3d):
    """ Update momentum to get the next-step values """
    pres_grad4u, pres_grad4v, pres_grad4w = pres_grad.pressure_gradient_force(pi0_now, pip_now, theta_now, x3d, y3d,
                                                                              z3d)

    u_next = u_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + (adv4u + pres_grad4u + fv) * nl.dt * 2.0
    v_next = v_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + (adv4v + pres_grad4v - fu) * nl.dt * 2.0
    w_next = w_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + (adv4w + pres_grad4w + b8w) * nl.dt * 2.0
    u_next3 = padding3_array(u_next)
    v_next3 = padding3_array(v_next)
    w_next3 = padding3_array(w_next)
    w_next3 = bc.set_w_bc(w_next3)
    return u_next3, v_next3, w_next3


def asselin_filter(u_prev, v_prev, w_prev, rho0_theta0_prev, theta_prev, qv_prev,
                   u_now, v_now, w_now, rho0_theta0_now, theta_now, qv_now,
                   u_next, v_next, w_next, rho0_theta0_next, theta_next, qv_next, z3d4w):
    """ Asselin filter """
    u_now_f = (1.0 - 2.0 * nl.asselin_r) * u_now + nl.asselin_r * (u_prev + u_next)
    v_now_f = (1.0 - 2.0 * nl.asselin_r) * v_now + nl.asselin_r * (v_prev + v_next)
    w_now_f = (1.0 - 2.0 * nl.asselin_r) * w_now + nl.asselin_r * (w_prev + w_next)
    rho0_theta0_now_f = (1.0 - 2.0 * nl.asselin_r) * rho0_theta0_now + nl.asselin_r * (
            rho0_theta0_prev + rho0_theta0_next)
    theta_now_f = (1.0 - 2.0 * nl.asselin_r) * theta_now + nl.asselin_r * (theta_prev + theta_next)
    qv_now_f = (1.0 - 2.0 * nl.asselin_r) * qv_now + nl.asselin_r * (qv_prev + qv_next)

    pi0_now_f = (rho0_theta0_now_f * nl.Rd / nl.p00) ** (nl.Rd / nl.Cv)
    pi0_now_f8w = pres_eqn.interpolate_scalar2w(pi0_now_f)
    d_pi0_dz = (pi0_now_f8w[:, :, 1:] - pi0_now_f8w[:, :, 0:-1]) / (
            z3d4w[:, :, nl.ngz + 1:-nl.ngz] - z3d4w[:, :, nl.ngz:-(nl.ngz + 1)])
    theta0_now_f_part = -nl.g / nl.Cp / d_pi0_dz
    x_size, y_size, z_size = theta0_now_f_part.shape
    bottom = jnp.reshape(theta0_now_f_part[:, :, 0], (x_size, y_size, 1))
    top = jnp.reshape(theta0_now_f_part[:, :, -1], (x_size, y_size, 1))
    theta0_now_f = jnp.concatenate((bottom, theta0_now_f_part, top), axis=2)

    rho0_now_f = rho0_theta0_now_f / theta0_now_f

    return u_now_f, v_now_f, w_now_f, rho0_theta0_now_f, rho0_now_f, theta0_now_f, theta_now_f, qv_now_f


def prep_momentum_eqn(rho0, u, v, w, flow_divergence, u_sfc_flux, v_sfc_flux, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w):
    """ Calculate advection tendencies for momentum equations """
    weps = 1.0e-16  # set based on CM1 subroutines
    adv4u = adv.advection_u(rho0, u, v, w, weps, flow_divergence, u_sfc_flux, x3d, y3d4v, z3d4w)
    adv4v = adv.advection_v(rho0, u, v, w, weps, flow_divergence, v_sfc_flux, x3d4u, y3d, z3d4w)
    adv4w = adv.advection_w(rho0, u, v, w, weps, flow_divergence, x3d4u, y3d4v, z3d)
    return adv4u, adv4v, adv4w


def compute_rho0_theta0_tendency(rho0_theta0, rho0, u, v, w, heating, x3d4u, y3d4v, z3d4w):
    """ Compute the tendency for rho0*theta0 """
    weps = 1.0e-17  # chosen based on CM1 weps for theta equation
    convergence = adv.rho0_theta0_flux_convergence(rho0_theta0, u, v, w, weps, x3d4u, y3d4v, z3d4w)
    # heating term here has the unit of K/s. It should be the heating tendency for theta equation, i.e., Hm/Cp/pi0
    rho0_theta0_heating = rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * heating
    tendency = convergence + rho0_theta0_heating  # no ghost points tendency
    return tendency, rho0_theta0_heating


def compute_theta_tendency(rho0, theta, u, v, w, flow_divergence, sfc_flux, heating, x3d4u, y3d4v, z3d4w):
    """ Compute the tendency for theta """
    weps = 1.0e-17  # chosen based on CM1 weps for theta equation
    adv_theta = adv.advection_scalar(rho0, theta, u, v, w, weps, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w)
    tendency = adv_theta + heating
    return tendency


def compute_qv_tendency(rho0, qv, u, v, w, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w):
    """ Compute the tendency for qv """
    weps = 1.0e-20  # chosen based on CM1 weps for qv equation
    adv_qv = adv.advection_scalar(rho0, qv, u, v, w, weps, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w)
    tendency = adv_qv  # + microphysics in the future
    return tendency


def get_heating(theta, theta0_ic):
    """ Compute the heating rate for theta

    In a full complexity LES, heating should include the effects of radiation and microphysics. Here we use Newtonian
    relaxation for the time being.
    """
    heating = rad.newton_radiation(theta, theta0_ic)  # unit: K/s
    # heating here is Hm/Cp/pi0, the tendency of theta due to heating, where Hm has the unit J/s/kg
    return heating


def space_integration(scalar, x3d4u, y3d4v, z3d4w):
    """ Compute the spatial integration of a scalar """
    dx = x3d4u[nl.ngx+1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] - x3d4u[nl.ngx:-(nl.ngx+1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
    dy = y3d4v[nl.ngx:-nl.ngx, nl.ngy+1:-nl.ngy, nl.ngz:-nl.ngz] - x3d4u[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz]
    dz = z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] - x3d4u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)]
    return jnp.sum(scalar[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * dx * dy * dz)


def get_rho(pi0, pip, theta, qv):
    """ Compute density """
    pi = pi0 + pip
    t = pi * theta    # temperature
    p = pi**(nl.Cp / nl.Rd) * nl.p00
    rho = p / nl.Rd / ((1.0 + nl.repsm1*qv) * t)
    return rho


def padding3_array(arr):
    """ Padding an array with three ghost points on each side """
    arr_x = jnp.concatenate((arr[-nl.ngx:, :, :], arr, arr[0:nl.ngx, :, :]), axis=0)
    arr_xy = jnp.concatenate((arr_x[:, -nl.ngx:, :], arr_x, arr_x[:, 0:nl.ngx, :]), axis=1)
    arr_xyz = jnp.concatenate((arr_xy[:, :, 0], arr_xy, arr_xy[:, :, -1]), axis=2)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.
    return arr_xyz
