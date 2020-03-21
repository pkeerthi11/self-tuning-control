import numpy as np
import simulation as sim
import scipy.signal as signal


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


class ProportionalControllerWithHistory(Controller):

    def __init__(self, gain, tt, omega=0.01, eq=None):
        self.gain = gain
        self.w = 0
        self.omega = omega
        self.dt = tt[1]-tt[0]
        self.e_history = np.zeros(tt.shape)
        self.w_history = np.zeros(tt.shape)
        self.c_history = np.zeros(tt.shape)
        self.index = 0
        self.eq = eq

    def _update_w(self, state):
        if self.eq is None:
            self.w += self.omega * self.error_signal(state) * self.dt

    def error_signal(self, state):
        if self.eq is None:
            return state[0] - self.w
        else:
            return state[0] - self.eq

    def __call__(self, history):
        state = history[-1]
        self.w_history[self.index] = self.w
        self.e_history[self.index] = self.error_signal(state)
        self.index += 1
        input_signal = -self.gain * self.error_signal(state)
        self._update_w(state)
        return input_signal, 0


class ProportionalAdaptiveControllerWithHistory(Controller):

    def __init__(self, sigma, tau_theta, tt, omega=0.01, eq=None):
        self.sigma = sigma
        self.tau_theta = tau_theta
        self.w = 0
        self.omega = omega
        self.dt = tt[1]-tt[0]
        self.e_history = np.zeros(tt.shape)
        self.w_history = np.zeros(tt.shape)
        self.c_history = np.zeros(tt.shape)
        self.index = 0
        self.eq = eq

    def _update_w(self, state):
        if self.eq is None:
            self.w += self.omega * self.error_signal(state) * self.dt

    def error_signal(self, state):
        if self.eq is None:
            return state[0] - self.w
        else:
            return state[0] - self.eq

    def grad_theta(self, state):
        grad = (abs(self.error_signal(state)) - self.sigma * state[2]) / self.tau_theta
        return grad

    def __call__(self, history):
        state = history[-1]
        gain = state[2]
        self.w_history[self.index] = self.w
        self.e_history[self.index] = self.error_signal(state)
        input_signal = -gain * self.error_signal(state)
        self.c_history[self.index] = input_signal
        self.index += 1
        self._update_w(state)
        return input_signal, self.grad_theta(state)


class ProportionalIntegralController(Controller):
    def __init__(self, proportional_gain, integral_gain, dt, omega=0.1):
        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain
        self.integral_term = 0
        self.w = 0
        self.omega = omega
        self.dt = dt

    def _update_w(self, state):
        self.w += self.omega * self.error_signal(state) * self.dt

    def error_signal(self, state):
        return state[0] - self.w

    def __call__(self, history):
        state = history[-1]
        e = self.error_signal(state)
        integral_term = -(self.integral_term + self.integral_gain * e * self.dt)
        proportional_term = -self.proportional_gain * e
        input_signal = integral_term + proportional_term
        self._update_w(state)
        self.integral_term = integral_term
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
        input = -gain * self.error_signal(state)
        self._update_w(state)
        return input, self.grad_theta(history)


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
        # fs = 100/self.dt
        # nyq = fs / 2
        # low = 10 / nyq
        # hig = 30 / nyq
        # b, a = signal.cheby1(5, 0.1, [low, hig], 'bandpass')
        # f_s10 = signal.filtfilt(b, a, s10)
        e = np.ptp(sim.butter_bandpass_filter(s10, 15, 30, 100/self.dt, 5))
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
            input = -gain * (last_state[0] - self.w)
            self._update_w(last_state)
            return input, self.grad_theta(history)


class AdaptiveControllerDeadband(Controller):
    def __init__(self, sigma, tau_theta, lam, mu_min, mu_max, tail_len=5000):
        self.lam = lam
        self.sigma = sigma
        self.tau_theta = tau_theta
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.tail_len = tail_len

    def error_signal(self, history):
        x1 = history[:, 0]
        e = np.ptp(x1[-self.tail_len:])
        return e

    def grad_theta(self, history):
        state = history[-1]
        grad = (rect(self.error_signal(history) - self.lam) - self.sigma * state[2]) / self.tau_theta
        return grad

    def __call__(self, history):
        current_state = history[-1]
        proportional_signal = current_state[0] * current_state[2]
        if self.error_signal(history) < self.lam:
            stim = 0
        elif proportional_signal < self.mu_max:
            stim = -max(self.mu_min, proportional_signal)
        else:
            stim = -self.mu_max
        return stim, self.grad_theta(history)
