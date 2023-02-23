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

plt.rc('font', size=6.)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=6.)
plt.rc('ytick', labelsize=6.)
plt.rc('axes', labelsize=6.)
plt.rc('axes', linewidth=0.5)

plt.rcParams["errorbar.capsize"]

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=[5., 3.])

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
            r1 = 9.115528166294098
            r2 = 11.672483086586
            circle1 = plt.Circle((0., 0.), r1, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle2 = plt.Circle((0., 0.), r2, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            ax[i][j].set_ylabel('Y [au]')
            ax[i][j].add_patch(circle1)
            ax[i][j].add_patch(circle2)
        elif j == 1:
            r1 = 18.479355573654175
            r2 = 19.05549168586731
            r3 = 22.217072248458862
            r4 = 25.120029449462894
            circle1 = plt.Circle((0., 0.), r1, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle2 = plt.Circle((0., 0.), r2, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle3 = plt.Circle((0., 0.), r3, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle4 = plt.Circle((0., 0.), r4, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            ax[i][j].add_patch(circle1)
            ax[i][j].add_patch(circle2)
            ax[i][j].add_patch(circle3)
            ax[i][j].add_patch(circle4)
        else:
            r1 = 27.772419452667236
            r2 = 36.23497247695923
            r3 = 38.32187294960023
            circle1 = plt.Circle((0., 0.), r1, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle2 = plt.Circle((0., 0.), r2, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            circle3 = plt.Circle((0., 0.), r3, color='g', fill=False,
                                 linewidth=0.6, linestyle='--', alpha=0.3)
            ax[i][j].add_patch(circle1)
            ax[i][j].add_patch(circle2)
            ax[i][j].add_patch(circle3)

        if i == 1:
            ax[i][j].set_xlabel('X [au]')

fig.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax, label=r'Brigthness [Jy/px]')
plt.savefig(fname='Fig4.png', dpi=400)
