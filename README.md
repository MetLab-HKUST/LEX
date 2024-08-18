## LEX: Large-Eddy simulation in JAX

This is the LEX model website. LEX is a Large-Eddy simulation model written in JAX. It can be used for hybrid simulations combining machine learning and physics-based numerical simulations. 

LES used a pseudo-incompressible dynamical core. It uses the Weighted Essentially Nonoscillatory (WENO) schemes to calculate fluxes and a third-order Strong Stability Preserving Runge-Kutta (SSPRK3) to integrate in time. The following figure is our benchmark simulation of a simple warm bubble case. Its accuracy is close to that of the fully compressible Cloud Model 1 (CM1) for this simple case.

![Warm Bubble Case](initial_comparison/theta_perturbation_LEX_SSPRK3_dx100dt10_dx600dt30.jpg)


