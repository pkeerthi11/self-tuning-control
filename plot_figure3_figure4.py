import math
import pickle

import matplotlib.pyplot as plt
import numpy as np


def plot_controller_comparison(filename, prefix, ylim=None):
    with open(filename, 'rb') as f:
        res = pickle.load(f)

    constants = res['constants']
    max_delay = max(constants[6:10])
    simulation_time = res['simulation_time']
    dt = res['dt']
    timestop = simulation_time
    ts = math.floor((max_delay + timestop) / dt)
    tt = np.arange(-max_delay, timestop, dt) / 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('Proportional control')
    ax1.plot(tt, res['proportional'][:ts, 0])
    ax1.plot(tt, res['proportional'][:ts, 1])
    ax1.legend(['STN', 'GPe'])
    ax1.set_xlim([0, 3])
    if ylim is not None:
        ax1.set_ylim(ylim[0])
    ax1.axvline(0.2, color='k', linestyle='--', linewidth=2)
    ax1.axvline(0.75, color='k', linestyle='--', linewidth=2)
    ax2.set_title('Self-tuning control')
    ax2.plot(tt, res['adaptive'][:ts, 0])
    ax2.plot(tt, res['adaptive'][:ts, 1])
    ax2.legend(['STN', 'GPe'])
    ax2.axvline(0.2, color='k', linestyle='--', linewidth=2)
    ax2.axvline(0.75, color='k', linestyle='--', linewidth=2)
    ax2.set_xlim([0, 3])
    if ylim is not None:
        ax2.set_ylim(ylim[0])
    plt.xlabel('Time [s]')
    fig.savefig('plots/' + prefix + '_1.png', bbox_inches='tight')
    fig.savefig('plots/svg/' + prefix + '_1.svg', bbox_inches='tight')

    fig2, (ax4, ax5) = plt.subplots(2, 1, sharex=True)
    fig2.suptitle('Theta')
    ax4.set_title('Proportional control')
    ax4.plot(tt, res['proportional'][:ts, 2])
    ax4.axvline(0.2, color='k', linestyle='--', linewidth=2)
    ax4.axvline(0.75, color='k', linestyle='--', linewidth=2)
    ax4.set_xlim([0, 3])
    ax5.set_title('Self-tuning control')
    ax5.plot(tt, res['adaptive'][:ts, 2])
    plt.xlabel('Time [s]')
    ax5.axvline(0.2, color='k', linestyle='--', linewidth=2)
    ax5.axvline(0.75, color='k', linestyle='--', linewidth=2)
    ax5.set_xlim([0, 3])
    fig2.savefig('plots/' + prefix + '_2.png', bbox_inches='tight')
    fig2.savefig('plots/svg/' + prefix + '_2.svg', bbox_inches='tight')

    fig3, (ax7, ax8) = plt.subplots(2, 1, sharex=True)
    iat = 100
    ialen = int(iat / dt)
    tt = np.arange(-max_delay + iat, timestop, dt) / 1000

    x0 = res['proportional'][:ts, 0]
    x1 = res['proportional'][:ts, 1]
    ia0 = np.array([np.ptp(x0[i:i + ialen]) for i in range(ts - ialen)])
    ia1 = np.array([np.ptp(x1[i:i + ialen]) for i in range(ts - ialen)])
    ax7.set_title('Instantaneous amplitude; proportional control')
    ax7.plot(tt, ia0)
    ax7.plot(tt, ia1)
    ax7.legend(['STN', 'GPe'])
    ax7.axvline(0.2, color='k', linestyle='--', linewidth=2)
    ax7.axvline(0.75, color='k', linestyle='--', linewidth=2)
    ax7.set_xlim([0, 3])
    if ylim is not None:
        ax7.set_ylim(ylim[1])
    x0 = res['adaptive'][:ts, 0]
    x1 = res['adaptive'][:ts, 1]
    ia0 = np.array([np.ptp(x0[i:i + ialen]) for i in range(ts - ialen)])
    ia1 = np.array([np.ptp(x1[i:i + ialen]) for i in range(ts - ialen)])
    ax8.set_title('Instantaneous amplitude; self-tuning control')
    ax8.plot(tt, ia0)
    ax8.plot(tt, ia1)
    ax8.legend(['STN', 'GPe'])
    ax8.axvline(0.2, color='k', linestyle='--', linewidth=2)
    ax8.axvline(0.75, color='k', linestyle='--', linewidth=2)
    ax8.set_xlim([0, 3])
    if ylim is not None:
        ax8.set_ylim(ylim[1])
    plt.xlabel('Time [s]')
    fig3.savefig('plots/' + prefix + '_3.png', bbox_inches='tight')
    fig3.savefig('plots/svg/' + prefix + '_3.svg', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_controller_comparison('simulation_results/results_figure_3', 'figure3', ylim=[[5, 99], [-1, 61]])
    plot_controller_comparison('simulation_results/results_figure_4', 'figure4', ylim=[[0, 300], [-1, 280]])
