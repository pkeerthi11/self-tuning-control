import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib


def single_plot(ax, f_range, ctx_range, data, input_amplitude, plot_color, alphas, grays, plot_loglog=True):
    leg = []

    for i in reversed(range(len(ctx_range))):
        if plot_color:
            if plot_loglog:
                ax.loglog(f_range, data[:, i, 1] / input_amplitude, color='#1f77b4', alpha=alphas[i])
                ax.loglog(f_range, data[:, i, 2] / input_amplitude, color='#1f77b4', alpha=alphas[i],
                           linestyle='--')
            else:
                ax.semilogx(f_range, data[:, i, 1] / input_amplitude, color='#1f77b4', alpha=alphas[i])
                ax.semilogx(f_range, data[:, i, 2] / input_amplitude, color='#1f77b4', alpha=alphas[i],
                          linestyle='--')
        else:
            if plot_loglog:
                ax.loglog(f_range, data[:, i, 1] / input_amplitude, color=grays[i])
                ax.loglog(f_range, data[:, i, 2] / input_amplitude, color=grays[i], linestyle='--')
            else:
                ax.semilogx(f_range, data[:, i, 1] / input_amplitude, color=grays[i])
                ax.semilogx(f_range, data[:, i, 2] / input_amplitude, color=grays[i], linestyle='--')
        leg.append('ctx=%.1f $\pm$ %.1f - No Control' % (ctx_range[i], input_amplitude))
        leg.append('ctx=%.1f $\pm$ %.1f - Self-Tuning Controller' % (ctx_range[i], input_amplitude))
    ax.legend(leg)
    ax.set_xlim([3, 100])


def full_plot(f_range, ctx_range, stn_amplitude, gpe_amplitude, input_amplitude, plot_color, alphas, grays, plot_loglog):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.set_title('STN amplitude ratio')
    single_plot(ax1, f_range, ctx_range, stn_amplitude, input_amplitude, plot_color, alphas, grays, plot_loglog)

    ax2.set_title('GPe amplitude ratio')
    single_plot(ax2, f_range, ctx_range, gpe_amplitude, input_amplitude, plot_color, alphas, grays, plot_loglog)

    plt.xlabel('Frequency [Hz]')
    return fig


if __name__ == "__main__":
    input_amplitude = 10
    plot_color = False

    p = pathlib.Path('simulation_results/results_figure_2a')

    with open(p, 'rb') as results_file:
        res = pickle.load(results_file)
        stn_amplidute = res['stn']
        gpe_amplidute = res['gpe']
        f_range = res['f_range']
    ctx_range = np.array([10, 22])
    alphas = [0.6, 1]
    grays = ['0.5', '0.1']

    fig = full_plot(f_range, ctx_range, stn_amplidute, gpe_amplidute, input_amplitude, plot_color, alphas, grays,
                    plot_loglog=True)
    fig_loglin = full_plot(f_range, ctx_range, stn_amplidute, gpe_amplidute, input_amplitude, plot_color, alphas, grays,
                           plot_loglog=False)

    fig.savefig('plots/bode_plot')
    fig.savefig('plots/svg/bode_plot.svg')
    fig_loglin.savefig('plots/bode_plot_loglin')
    fig_loglin.savefig('plots/svg/bode_plot_loglin.svg')

    plt.close(fig)
    plt.close(fig_loglin)
