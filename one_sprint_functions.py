""" One sprint means many steps that do not need to be interrupted and after which we save the data to an output file """
import jax
import jax.numpy as jnp
from functools import partial
import one_step_integration as one
import namelist_n_constants as nl
import dl_models as dlm


@partial(jax.jit, static_argnames=['model_opt'])
def first_step_integration_ssprk3(phys_state, base_state, grids, model_opt):
    """ The first step using SSPRK3 method, mainly needed by the leapfrog scheme """
    int_opt = model_opt[0]
    xi0 = phys_state
    xi1, _, sfc_others, heating, sgs_tend = one.rk_sub_step0(xi0, xi0, base_state, grids, model_opt, nl.dt/2.0)
    if int_opt == 2:
        _, _, _, _, pip_now, _ = xi1
        xi2, _ = one.rk_sub_step_other(xi1, xi1, base_state, grids, heating, sfc_others, sgs_tend, model_opt, nl.dt/2.0)
        xi3, _ = one.rk_sub_step_other(xi2, xi2, base_state, grids, heating, sfc_others, sgs_tend, model_opt, nl.dt/2.0)
        xi3 = tuple(map(lambda x, y: 2.0/3.0*x + 1.0/3.0*y, xi0, xi3))
        xi4, _ = one.rk_sub_step_other(xi3, xi3, base_state, grids, heating, sfc_others, sgs_tend, model_opt, nl.dt/2.0)
        theta_next, u_next, v_next, w_next, _, qv_next = xi4
    # for SSPRK3, we only want to get surface flux for the initial time

    if int_opt == 2:    # leapfrog
        theta_now, u_now, v_now, w_now, _, qv_now = phys_state
        phys_state = (theta_now, theta_next, pip_now, qv_now, qv_next, u_now, u_next, v_now, v_next, w_now, w_next)
    # for SSPRK3, just output the I.C.

    return phys_state, sfc_others


@partial(jax.jit, static_argnames=['model_opt'])
def ssprk3_big_step(phys_state, base_state, grids, model_opt):
    """ Integration using SSPRK3 method for one big step

    Four-stage SSPRK3 based on Durran (2010), Page 56.
    """
    xi0 = phys_state
    xi1, _, sfc_others, heating, sgs_tend = one.rk_sub_step0(xi0, xi0, base_state, grids, model_opt, nl.dt/2.0)
    _, _, _, _, pip_now, _ = xi1
    xi2, _ = one.rk_sub_step_other(xi1, xi1, base_state, grids, heating, sfc_others, sgs_tend, model_opt, nl.dt/2.0)
    xi3, _ = one.rk_sub_step_other(xi2, xi2, base_state, grids, heating, sfc_others, sgs_tend, model_opt, nl.dt/2.0)
    xi3 = tuple(map(lambda x, y: 2.0/3.0*x + 1.0/3.0*y, xi0, xi3))
    xi4, _ = one.rk_sub_step_other(xi3, xi3, base_state, grids, heating, sfc_others, sgs_tend, model_opt, nl.dt/2.0)
    theta_next, u_next, v_next, w_next, _, qv_next = xi4
    phys_state = (theta_next, u_next, v_next, w_next, pip_now, qv_next)

    return phys_state, sfc_others


@partial(jax.jit, static_argnames=['model_opt'])
def ssprk3_sprint_dl_train(theta_now, u_now, v_now, w_now, pip_prev, qv_now, base_state, grids, model_opt, dl_params, scaling_params):
    """ Integration using SSPRK3 method for sprint_n big steps, with deep learning correction

    We want to get a vmap version of this function, so the input tuple phys_state is unpacked.
    """
    theta_stack = jnp.zeros((nl.sprint_n, nl.nx, nl.ny, nl.nz))
    pip_stack = jnp.zeros((nl.sprint_n, nl.nx, nl.ny, nl.nz))
    qv_stack = jnp.zeros((nl.sprint_n, nl.nx, nl.ny, nl.nz))
    u_stack = jnp.zeros((nl.sprint_n, nl.nx+1, nl.ny, nl.nz))
    v_stack = jnp.zeros((nl.sprint_n, nl.nx, nl.ny+1, nl.nz))
    w_stack = jnp.zeros((nl.sprint_n, nl.nx, nl.ny, nl.nz+1))

    theta_min, theta_max, u_min, u_max, v_min, v_max, w_min, w_max, qv_min, qv_max, _, _, _, _ = scaling_params

    for i in range(nl.sprint_n):
        # dynamical core integration
        phys_state_now = (theta_now, u_now, v_now, w_now, pip_prev, qv_now)
        theta_prev = theta_now
        phys_state_next, _ = ssprk3_big_step(phys_state_now, base_state, grids, model_opt)
        theta_now, u_now, v_now, w_now, pip_prev, qv_now = phys_state_next
        # DL corrections
        phys_state_stack = dlm.phys_state_tuple2array(phys_state_next, scaling_params)
        # This stack is the input for the DL model
        dl_correct = nl.dl_model.apply({'params': dl_params}, phys_state_stack).squeeze()
        # theta_correct, u_correct, v_correct, w_correct, pip_correct, qv_correct = dlm.correct_array2tuple(dl_correct)
        theta_correct, u_correct, v_correct, w_correct, qv_correct = dlm.correct_array2tuple(dl_correct)

        theta_correct = theta_correct * (theta_max - theta_min).squeeze()
        u_correct = u_correct * (u_max - u_min).squeeze()
        v_correct = v_correct * (v_max - v_min).squeeze()
        w_correct = w_correct * (w_max - w_min).squeeze()
        qv_correct = qv_correct * (qv_max - qv_min).squeeze()

        theta = theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + theta_correct
        u = u_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + u_correct
        v = v_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + v_correct
        w = w_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + w_correct
        # pip = pip_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + pip_correct
        # Maybe we shouldn't correct pi'
        qv = qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + qv_correct

        # theta = jnp.where(theta<300.0, 300.0, theta)
        # Avoid too low temperature, which may cause instability. Special for the rising thermal bubble test case
        # qv = jnp.where(qv<0.0, 0.0, qv)

        # Padding ghost points 
        theta_now = one.padding3_array(theta)
        u_now = one.padding3u_array(u)
        v_now = one.padding3v_array(v)
        w_now = one.padding3_array(w)
        # pip_prev = one.padding3_array(pip)
        # Maybe we shouldn't correct pi'
        qv_now = one.padding3_array(qv)

        # Save each step for computing loss
        theta_stack = theta_stack.at[i, :, :, :].set(theta)
        u_stack = u_stack.at[i, :, :, :].set(u)
        v_stack = v_stack.at[i, :, :, :].set(v)
        w_stack = w_stack.at[i, :, :, :].set(w)
        pip_stack = pip_stack.at[i, :, :, :].set(pip_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
        qv_stack = qv_stack.at[i, :, :, :].set(qv)

    return theta_stack, u_stack, v_stack, w_stack, pip_stack, qv_stack


@partial(jax.jit, static_argnames=['model_opt'])
def ssprk3_sprint_dl_train_vmap(theta_batch, u_batch, v_batch, w_batch, pip_batch, qv_batch,
                                base_state, grids, model_opt, dl_params, scaling_params):
    theta_stack, u_stack, v_stack, w_stack, pip_stack, qv_stack = jax.vmap(
        ssprk3_sprint_dl_train, (0, 0, 0, 0, 0, 0, None, None, None, None, None),
        (0, 0, 0, 0, 0, 0))(theta_batch, u_batch, v_batch, w_batch, pip_batch, qv_batch,
                            base_state, grids, model_opt, dl_params, scaling_params)
    return theta_stack, u_stack, v_stack, w_stack, pip_stack, qv_stack


@partial(jax.jit, static_argnames=['model_opt'])
def ssprk3_sprint(phys_state, base_state, grids, model_opt):
    """ Integration using SSPRK3 method for sprint_n big steps """
    for i in range(nl.sprint_n):
        phys_state, sfc_others = ssprk3_big_step(phys_state, base_state, grids, model_opt)

    return phys_state, sfc_others


@partial(jax.jit, static_argnames=['model_opt'])
def ssprk3_sprint_dl(phys_state, base_state, grids, model_opt, dl_params, scaling_params):
    """ Integration using SSPRK3 method for sprint_n big steps, with deep learning correction

    This is the application, not training. For training the DL model, use ssprk3_sprint_dl_train
    """
    theta_min, theta_max, u_min, u_max, v_min, v_max, w_min, w_max, qv_min, qv_max, _, _, _, _ = scaling_params
    phys_state_now = phys_state
    for i in range(nl.sprint_n):
        # dynamical core integration
        phys_state_next, sfc_others = ssprk3_big_step(phys_state_now, base_state, grids, model_opt)
        theta_now, u_now, v_now, w_now, pip_prev, qv_now = phys_state_next
        # DL corrections
        phys_state_stack = dlm.phys_state_tuple2array(phys_state_next, scaling_params)
        # This stack is the input for the DL model
        dl_correct = nl.dl_model.apply({'params': dl_params}, phys_state_stack).squeeze()
        # theta_correct, u_correct, v_correct, w_correct, pip_correct, qv_correct = dlm.correct_array2tuple(dl_correct)
        theta_correct, u_correct, v_correct, w_correct, qv_correct = dlm.correct_array2tuple(dl_correct)

        theta_correct = theta_correct * (theta_max - theta_min).squeeze()
        u_correct = u_correct * (u_max - u_min).squeeze()
        v_correct = v_correct * (v_max - v_min).squeeze()
        w_correct = w_correct * (w_max - w_min).squeeze()
        qv_correct = qv_correct * (qv_max - qv_min).squeeze()

        theta = theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + theta_correct
        u = u_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + u_correct
        v = v_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + v_correct
        w = w_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + w_correct
        # pip = pip_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + pip_correct
        # Maybe we shouldn't correct pi'
        qv = qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + qv_correct

        # theta = jnp.where(theta<300.0, 300.0, theta)
        # Avoid too low temperature, which may cause instability. Special for the rising thermal bubble test case
        # qv = jnp.where(qv<0.0, 0.0, qv)

        # Padding ghost points
        theta_now = one.padding3_array(theta)
        u_now = one.padding3u_array(u)
        v_now = one.padding3v_array(v)
        w_now = one.padding3_array(w)
        # pip_prev = one.padding3_array(pip)
        # Maybe we shouldn't correct pi'
        qv_now = one.padding3_array(qv)
        phys_state_now = (theta_now, u_now, v_now, w_now, pip_prev, qv_now)

    return phys_state_now, sfc_others


@partial(jax.jit, static_argnames=['model_opt'])
def leapfrog_sprint(phys_state, base_state, grids, model_opt):
    """ Integrate with leapfrog method for sprint_n steps

    Here one sprint is a number of consecutive steps, between which we don't need save data to an output file. After
    one sprint, we return to the main program and save data to a file
    """

    for i in range(nl.sprint_n):
        theta_prev, theta_now, pip_prev, qv_prev, qv_now, u_prev, u_now, v_prev, v_now, w_prev, w_now = phys_state

        phys_state_prev = (theta_prev, u_prev, v_prev, w_prev, pip_prev, qv_prev)
        phys_state_now = (theta_now, u_now, v_now, w_now, pip_prev, qv_now)

        phys_state_next, _, sfc_others, _ = one.rk_sub_step0(phys_state_prev, phys_state_now, base_state, grids, model_opt, 2.0*nl.dt)

        theta_next, u_next, v_next, w_next, pip_now, qv_next = phys_state_next

        # apply Asselin filter
        u_now, v_now, w_now, theta_now, qv_now = one.asselin_filter(
                        u_prev, v_prev, w_prev, theta_prev, qv_prev,
                        u_now, v_now, w_now, theta_now, qv_now,
                        u_next, v_next, w_next, theta_next, qv_next)

        phys_state = (theta_now, theta_next, pip_now, qv_now, qv_next, u_now, u_next, v_now, v_next, w_now, w_next)

    return phys_state, sfc_others
