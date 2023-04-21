"""
Script to generate Figure 4 of the paper.

Giovanni Picogna, 20.02.2023
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from radmc3dPy import image

import scienceplots

plt.style.use('science')

plt.rc('font', size=18.)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=18.)
plt.rc('ytick', labelsize=18.)
plt.rc('axes', labelsize=18.)
plt.rc('lines', linewidth=3.)

plt.rcParams["errorbar.capsize"]

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=[18., 12.])

path = "../data/"

fnames = ["image085", "image3"]
gap = [10, 20, 30]
dpc = 140.

os.chdir(path)

for i in range(2):

    for j in range(3):

        xlab = 'X [AU]'
        ylab = 'Y [AU]'
        im = image.readImage(fname=fnames[i]+str(gap[j])+'au.out')
        data = im.image[:, :, 0].T
        clipmax = np.log10(data.max())
        clipmin = np.log10(data[data > 0.].min())
        data = np.log10(data.clip(1e-90))
        data = data.clip(clipmin, clipmax)

        # Convert to Jy/pixel
        data += np.log10(im.sizepix_x * im.sizepix_y /
                         (dpc * const.pc.value)**2. * 1e23)

        x = im.x*u.cm.to(u.AU)
        y = im.y*u.cm.to(u.AU)

        im = ax[i][j].pcolormesh(x, y, data, cmap='inferno', vmin=-8, vmax=-2)
        ax[i][j].set_xlim(-75., 75.)
        ax[i][j].set_ylim(-75., 75.)
        if j == 0:
            r1 = 8.2
            r2 = 8.5
            r3 = 9.04
            r4 = 11.7
            circle1 = plt.Circle((0., 0.), r1, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle2 = plt.Circle((0., 0.), r2, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle3 = plt.Circle((0., 0.), r3, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle4 = plt.Circle((0., 0.), r4, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            ax[i][j].set_ylabel('Y [au]')
            ax[i][j].add_patch(circle1)
            ax[i][j].add_patch(circle2)
            ax[i][j].add_patch(circle3)
            ax[i][j].add_patch(circle4)
        elif j == 1:
            r1 = 17.11
            r2 = 18.48
            r3 = 19.06
            r4 = 22.37
            r5 = 25.12
            circle1 = plt.Circle((0., 0.), r1, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle2 = plt.Circle((0., 0.), r2, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle3 = plt.Circle((0., 0.), r3, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle4 = plt.Circle((0., 0.), r4, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle5 = plt.Circle((0., 0.), r5, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            ax[i][j].add_patch(circle1)
            ax[i][j].add_patch(circle2)
            ax[i][j].add_patch(circle3)
            ax[i][j].add_patch(circle4)
            ax[i][j].add_patch(circle5)
        else:
            r1 = 27.01
            r2 = 27.77
            r3 = 36.23
            r4 = 38.32
            r5 = 39.41
            circle1 = plt.Circle((0., 0.), r1, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle2 = plt.Circle((0., 0.), r2, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle3 = plt.Circle((0., 0.), r3, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle4 = plt.Circle((0., 0.), r4, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle5 = plt.Circle((0., 0.), r5, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            ax[i][j].add_patch(circle1)
            ax[i][j].add_patch(circle2)
            ax[i][j].add_patch(circle3)
            ax[i][j].add_patch(circle4)
            ax[i][j].add_patch(circle5)

        if i == 1:
            ax[i][j].set_xlabel('X [au]')

fig.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax, label=r'Brigthness [Jy/px]')
plt.savefig(fname='Fig4.png', dpi=400)
