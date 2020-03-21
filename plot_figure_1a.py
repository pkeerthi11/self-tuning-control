import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    fpath = pathlib.Path('simulation_results/results_figure_1a')
    with open(fpath, 'rb') as results_file:
        res = pickle.load(results_file)
    stn_amplitude = res['stn']
    gpe_amplitude = res['gpe']
    constants = res['constants']
    c12_range = res['c12_range']
    c21_range = res['c21_range']

    if len(c12_range) == 1:
        plt.plot(c21_range, stn_amplitude[0], c21_range, gpe_amplitude[0])
        plt.legend(['STN', 'GPe'])
        plt.xlabel('c21')
        plt.show()
    else:
        stn_map = np.flip(stn_amplitude, 0)
        gpe_map = np.flip(gpe_amplitude, 0)
        ext = [min(c21_range), max(c21_range), 0, abs(min(c12_range))]
        asp = (max(c21_range) - min(c21_range)) / abs(min(c12_range))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
        im1 = ax1.imshow(stn_map, cmap=plt.cm.magma, extent=ext)
        ax1.set_title('STN')

        im2 = ax2.imshow(gpe_map, cmap=plt.cm.magma, extent=ext)
        ax2.set_title('GPe')

        for a in [ax1, ax2]:
            a.set_aspect(asp)
            a.set_ylabel('Connectivity from GPe to STN (c12)')
            a.set_xlabel('Connectivity from STN to GPe (c21)')

        fig.savefig('plots/amplitude_map')
        fig.savefig('plots/svg/amplitude_map.svg')
