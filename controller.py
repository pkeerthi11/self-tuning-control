import numpy as np
import scipy.signal as signal

import simulation as sim


def rect(x):
    return np.piecewise(x, [x > 0], [x, 0])


class Controller:

    def grad_theta(self, history):
        return 0


class ZeroController(Controller):
    def __init__(self):
        self.w = 0

    def error_signal(self, state):
        return 0

    def __call__(self, *args, **kwargs):
        return 0, 0


# class ProportionalController(Controller):
#     def __init__(self, gain, dt, omega=0.01):
#         self.gain = gain
#         self.w = 0
#         self.omega = omega
#         self.dt = dt

#     def _update_w(self, state):
#         self.w += self.omega * self.error_signal(state) * self.dt

#     def error_signal(self, state):
#         return state[0] - self.w

#     def __call__(self, history):
#         state = history[-1]
#         self._update_w(state)
#         input_signal = -self.gain * self.error_signal(state)
#         return input_signal, 0

class ProportionalController(Controller):
    def __init__(self, gain, dt, omega=0.01):
        self.gain = gain
        self.w = 0
        self.omega = omega
        self.dt = dt

    def _update_w(self, state):
        self.w += self.omega * self.error_signal(state) * self.dt

    def error_signal(self, state):
        return state[0] - self.w

    def __call__(self, history):
        state = history[-1]
        self._update_w(state)
        input_signal = -self.gain * self.error_signal(state)
        return input_signal, 0




class AdaptiveController(Controller):
    def __init__(self, sigma, tau_theta, dt, tail_len=500, omega=0.1):
        self.sigma = sigma
        self.tau_theta = tau_theta
        self.tail_len = tail_len
        self.w = 0
        self.omega = omega
        self.dt = dt

    def _update_w(self, state):
        self.w += self.omega * self.error_signal(state) * self.dt

    def error_signal(self, state):
        return state[0] - self.w

    def grad_theta(self, history):
        state = history[-1]
        grad = (abs(self.error_signal(state)) - self.sigma * state[2]) / self.tau_theta
        return grad

    def __call__(self, history):
        state = history[-1]
        gain = state[2]
        control_input = -gain * self.error_signal(state)
        self._update_w(state)
        return control_input, self.grad_theta(history)
    

class AdaptiveControllerFilter(AdaptiveController):
    def __init__(self, sigma, tau_theta, dt, tail_len=500, omega=0.1, deadzone=0):
        super().__init__(sigma, tau_theta, dt, tail_len, omega)
        self.samples = int(self.tail_len / self.dt)
        self.deadzone = deadzone

    def grad_theta(self, history):
        state = history[-self.samples:]
        last_state = state[-1]
        grad = (abs(self.error_signal(state)) - self.sigma * last_state[2]) / self.tau_theta
        return grad

    def error_signal(self, state):
        s = state[:, 0]
        s10 = signal.decimate(s, 10)
        e = np.ptp(sim.butter_bandpass_filter(s10, 15, 30, 100 / self.dt, 5))
        if e < self.deadzone:
            return 0
        else:
            return e

    def _update_w(self, last_state):
        self.w += self.omega * (last_state[0] - self.w) * self.dt

    def __call__(self, history):
        last_state = history[-1]
        if len(history) < self.samples:
            self._update_w(last_state)
            return 0, 0
        else:
            gain = last_state[2]
            control_input = -gain * (last_state[0] - self.w)
            self._update_w(last_state)
            return control_input, self.grad_theta(history)


class MemoryLessController(Controller):
    def __init__(self, gain, betas, dt, omega=0.01):
        self.betas = betas
        self.gain = gain
        self.w = 0
        self.omega = omega
        self.dt = dt

    def _update_w(self, state):
        self.w += self.omega * self.error_signal(state) * self.dt

    def error_signal(self, state):
        return state[0] - self.w

    def __call__(self, history):
        state = history[-1]
        self._update_w(state)
        input_signal = -self.gain**2 * self.betas[0] * state[0] - self.gain * self.betas[0] * state[0] 
        return input_signal, 0