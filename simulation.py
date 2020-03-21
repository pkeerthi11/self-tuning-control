from math import floor

import numpy as np
import scipy.signal as signal

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


# TODO: setup the activation functions
def activation1(x, constants):
    return sigmoid(x, constants[10], constants[11])


def activation2(x, constants):
    return sigmoid(x, constants[12], constants[13])


def single_simulation(constants, simulation_time, dt, control_mechanism, init_theta=40, mid_increase=0,
                      mid_increase_amplitude=0, steady_state_pad=0):
    max_delay = max(constants[6:10])
    tt = np.arange(-max_delay, simulation_time + steady_state_pad, dt)
    history = np.zeros((len(tt), 3))
    control_history = np.zeros((len(tt),))
    input_history = np.zeros((len(tt),))

    # TODO: setup the initial state
    hlen = int(floor(max_delay / dt))
    history[0:hlen + 1, :2] = 20
    history[0:hlen + 1, 2] = init_theta

    d11 = int(floor(constants[6] / dt))
    d12 = int(floor(constants[7] / dt))
    d21 = int(floor(constants[8] / dt))
    d22 = int(floor(constants[9] / dt))
    for i, t in enumerate(tt):
        if t <= 0:
            continue
        state = history[i - 1]

        control1, grad_theta = control_mechanism(history[:i, :])
        if t < 200 + steady_state_pad:
            control1, grad_theta = (0, 0)

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

    return history
