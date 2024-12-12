# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:19:20 2024

@author: pkeer
"""

import pickle

import controller as ctrl
import simulation as sim
import os
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    steady_state_pad = 1000

    constants = sim.constants_nevado_holgado_healthy
    constants[5] = -0.9
    constants[16] = 10
    constants[17] = 20
    constants[18] = 50

    simulation_time = 3000
    dt = 0.01

    print('Simulations')
    sigma = 0.01
    mi = (750, 10, 50)
    it = [20, 20, 0]
    tau_theta = 5
    prop_theta = 25
    gain_L = 4 #Increasing gain L results reduction in amplitude

    z_controller = ctrl.ZeroController()
    p_controller = ctrl.ProportionalController(gain=prop_theta, dt=dt)
    a_controller = ctrl.AdaptiveController(sigma=sigma, tau_theta=tau_theta, dt=dt)
    m_controller = ctrl.MemoryLessController(gain = gain_L, betas = [3,2], dt=dt)
    
    
    
    history_no = sim.single_simulation(constants, simulation_time, dt, z_controller, init_state=it, mid_increase=mi,
                                       steady_state_pad=steady_state_pad)
    
    history_adaptive = sim.single_simulation(constants, simulation_time, dt, control_mechanism=a_controller,
                                             control_start=200, init_state=it, mid_increase=mi,
                                             steady_state_pad=steady_state_pad)
    
    history_proportional = sim.single_simulation(constants, simulation_time, dt, control_mechanism=p_controller,
                                                 control_start=200, init_state=it, mid_increase=mi,
                                                 steady_state_pad=steady_state_pad)
    
    history_memoryless = sim.single_simulation(constants, simulation_time, dt, control_mechanism=m_controller,
                                             control_start=200, init_state=it, mid_increase=mi,
                                             steady_state_pad=steady_state_pad)
    
   
    times = dt*np.arange(int(simulation_time/dt))
    
    (fig,ax)=plt.subplots(2,2, sharey=True)
    ax[0,0].plot(times, history_no[-len(times):,0], label='STN')
    ax[0,1].plot(times, history_proportional[-len(times):,0], label='STN')
    ax[1,0].plot(times, history_adaptive[-len(times):,0], label='STN')
    ax[1,1].plot(times, history_memoryless[-len(times):,0], label='STN')
    ax[0,1].set_xlabel('Time (ms)')
    ax[0,0].set_ylabel('Firing Rate')
    
    ax[0,0].plot(times, history_no[-len(times):,1], label='GPe')
    ax[0,1].plot(times, history_proportional[-len(times):,1], label='GPe')
    ax[1,0].plot(times, history_adaptive[-len(times):,1], label='GPe')
    ax[1,1].plot(times, history_memoryless[-len(times):,1], label='GPe')
    ax[1,1].set_xlabel('Time (ms)')
    ax[1,0].set_ylabel('Firing Rate')
    
    ax[0,0].axvline(250, color = 'k', linestyle = '--') #, label='Controller On')
    ax[0,1].axvline(250, color = 'k', linestyle = '--') #, label='Controller On')
    ax[1,0].axvline(250, color = 'k', linestyle = '--') #, label='Controller On')
    ax[1,1].axvline(250, color = 'k', linestyle = '--') #, label='Controller On')
    
    ax[0,0].axvline(750, color = 'k', linestyle = '--') #, label='Controller On')
    ax[0,1].axvline(750, color = 'k', linestyle = '--') #, label='Controller On')
    ax[1,0].axvline(750, color = 'k', linestyle = '--') #, label='Controller On')
    ax[1,1].axvline(750, color = 'k', linestyle = '--') #, label='Controller On')
    
    
    
    
    ax[0,0].set_title('No controller')
    ax[0,1].set_title('Single-site Proportional Control (Fleming)')
    ax[1,0].set_title('Adaptive single-site Proportional Control (Fleming)')
    ax[1,1].set_title('Dual-site Proportional Control (Xia)')
    ax[1,1].legend(loc="upper right")

    print('Saving simulation results')
    file_dir = r"C:\Users\pkeer\OneDrive\Documents\GradSchool\Courses\Analysis of Nonlinear Dynamical Systems\FinalProject"
    fullfile = os.path.join(file_dir, 'simulation_results/results_figure_4')

    print('Saving simulation results')
    with open(fullfile, 'wb+') as f:
        pickle.dump({
            'proportional': history_proportional[int(steady_state_pad / dt):],
            'adaptive': history_adaptive[int(steady_state_pad / dt):],
            'constants': constants,
            'sigma': sigma,
            'mid_increase': mi,
            'initial_theta': it,
            'simulation_time': simulation_time,
            'dt': dt,
            'tau_theta': tau_theta,
            'prop_theta': prop_theta
        }, f)
