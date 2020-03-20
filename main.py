from math import floor, ceil
import scipy.signal as signal
import pathlib
import datetime

import matplotlib.pyplot as plt
import numpy as np

constants_nevado_holgado_healthy = [
    6,  # 0:  tau1 [ms]
    14,  # 1:  tau2 [ms]
    0,  # 2:  c11
    -1.12,  # 3:  c12
    19,  # 4:  c21
    -6.6,  # 5:  c22
    0,  # 6:  d11
    6,  # 7:  d12 [ms]
    6,  # 8:  d21 [ms]
    4,  # 9:  d22 [ms]
    300,  # 10: m1 [spk/s]
    17,  # 11: b1 [spk/s]
    400,  # 12: m2 [spk/s]
    75,  # 13: b2 [spk/s]
    2.42,  # 14: cctx
    -15.1,  # 15: cstr
    0,  # 16: oscillating input amplitude
    80,  # 17: oscillating input frequency
    27,  # 18: ctx level [spk/s]
    2  # 19: str level [spk/s]
]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def sigmoid(x, m, b):
    y = m / (1 + np.exp(-4 * x / m) * (m - b) / b)
    return y


def plot_history(filename, filename_svg, tt, history, control_history, input_history=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(10, 8))
    ax1.plot(tt, history[:, 0])
    ax1.plot(tt, history[:, 1])
    ax1.set_ylabel('Firing rate [spk/s]')
    ax1.legend(['STN', 'GPe'])
    # ax1.set_ylim([0, 260])
    ax2.plot(tt, control_history)
    ax2.legend(['$\mu(t)$'])
    ax3.plot(tt, history[:, 2])
    ax3.legend(['$\\theta(t)$'])
    plt.xlabel('Time [ms]')
    fig.savefig(filename)
    fig.savefig(filename_svg)

    if input_history is not None:
        fig2 = plt.figure(figsize=(10, 8))
        plt.plot(tt, input_history)
        plt.savefig('bode_plot_input')
    plt.close(fig)
    plt.close(fig2)


# TODO: setup the activation functions
def activation1(x, constants):
    return sigmoid(x, constants[10], constants[11])


def activation2(x, constants):
    return sigmoid(x, constants[12], constants[13])


def single_simulation(constants, max_delay, simulation_time, dt, control_mechanism, outdir='.', filename='test_image',
                      init_theta=40, plot=True, mid_increase=0, swap_constants=None, mid_increase_amplitude=0,
                      steady_state_pad=0):
    tt = np.arange(-max_delay, simulation_time + steady_state_pad, dt)
    history = np.zeros((len(tt), 3))
    control_history = np.zeros((len(tt),))
    input_history = np.zeros((len(tt),))

    if swap_constants is not None:
        swap_t_list = swap_constants[0]
        swap_index = 0

        # TODO: setup the initial state
    hlen = int(floor(max_delay / dt))
    history[0:hlen+1, :2] = 20
    history[0:hlen+1, 2] = init_theta

    d11 = int(floor(constants[6] / dt))
    d12 = int(floor(constants[7] / dt))
    d21 = int(floor(constants[8] / dt))
    d22 = int(floor(constants[9] / dt))
    for i, t in enumerate(tt):
        if t <= 0:
            continue
        if swap_constants is not None:
            if swap_index < len(swap_t_list) and t >= swap_t_list[swap_index]:
                constants = swap_constants[1][swap_index]
                if len(swap_constants) == 3 and hasattr(control_mechanism, 'eq') and control_mechanism.eq is not None:
                    control_mechanism.eq = swap_constants[2][swap_index]
                swap_index += 1
        state = history[i - 1]
        control1, grad_theta = control_mechanism(history[:i, :])
        if t < 200 + steady_state_pad:
            control1, grad_theta = (0, 0)
            # if control_mechanism == adaptive_control or control_mechanism == adaptive_filter_control:
            #     control1, grad_theta = control_mechanism(history[:i, :], sigma=sigma, tau_theta=tau_theta)
            # # elif control_mechanism == adaptive_filter_control:
            # #     control1, grad_theta = control_mechanism(history[:i, :], sigma=sigma)
            # else:
            #     control1, grad_theta = control_mechanism(history[:i, :])

        ampl_boost = 0
        if mid_increase_amplitude > 0 and t > 750 + steady_state_pad:
            ampl_boost = mid_increase_amplitude

        amplitude = constants[16] + ampl_boost
        f = constants[17]
        ctx_level = constants[18]
        str_level = constants[19]
        oscillating_input = amplitude * np.sin(2 * np.pi * f * t / 1000)

        str_input = str_level
        ctx_input = ctx_level + oscillating_input
        if mid_increase > 0:
            if t > 750 + steady_state_pad:
                ctx_input += mid_increase
        # TODO: calculate the inputs in a smart way
        inputs1 = constants[2] * history[i - 1 - d11, 0] + \
                  constants[3] * history[i - 1 - d12, 1] + constants[14] * ctx_input
        inputs2 = constants[4] * history[i - 1 - d21, 0] + \
                  constants[5] * history[i - 1 - d22, 1] + constants[15] * str_input
        grad = np.array([
            1 / constants[0] * (-state[0] + activation1(inputs1 + control1, constants)),
            1 / constants[1] * (-state[1] + activation2(inputs2, constants)),
            grad_theta
        ])
        history[i] = history[i - 1] + grad * dt
        control_history[i] = control1
        input_history[i] = oscillating_input

    if plot:
        if not isinstance(outdir, pathlib.Path):
            p = pathlib.Path(outdir)
        else:
            p = outdir
        f = p / filename
        f_svg = (p / 'svg' / filename).with_suffix('.svg')
        plot_history(f, f_svg, tt, history, control_history, input_history)
        plt.close()

    return history


def create_results_dir(suffix=''):
    s = '_' + suffix if suffix is not '' else ''
    dirname = 'simulations_%s%s' % (datetime.datetime.strftime(datetime.datetime.now(), '%y%m%d%H%M%S'), s)
    p = pathlib.Path(dirname)
    p.mkdir()
    p_svg = p / 'svg'
    p_svg.mkdir()
    return p


if __name__ == "__main__":
    constants = constants_nevado_holgado_healthy
    constants[4] = 20
    constants[3] = -10
    constants[5] = -0.9
    constants[14] = 5
    constants[15] = -139.4

    max_delay = max(constants[6:10])
    simulation_time = 3000
    dt = 0.005

    print('Simulations')
    tt = np.arange(-max_delay, simulation_time, dt)
    s = 0.05
    mi = 7.5
    it = 0

    # single_simulation(constants, max_delay, simulation_time, dt, control_mechanism=proportional_control, filename='proportional')
    # single_simulation(constants, max_delay, simulation_time, dt, control_mechanism=zero_control, filename='zero', init_theta=0)
    # history_proportional = single_simulation(constants, max_delay, simulation_time, dt,
    #                                          control_mechanism=proportional_control, filename='proportional',
    #                                          init_theta=0.9, mid_increase=mi)
    # history_adaptive = single_simulation(constants, max_delay, simulation_time, dt,
    #                                      control_mechanism=adaptive_control, filename=('adaptive_sigma_%.3f.png' % s),
    #                                      sigma=s, init_theta=it, mid_increase=mi)

    # with open('results_figure_3', 'wb+') as f:
    #     pickle.dump({
    #         'proportional': history_proportional,
    #         'adaptive': history_adaptive,
    #         'constants': constants,
    #         'sigma': s,
    #         'mid_increase': mi,
    #         'initial_theta': it,
    #         'simulation_time': simulation_time,
    #         'dt': dt
    #     }, f)

