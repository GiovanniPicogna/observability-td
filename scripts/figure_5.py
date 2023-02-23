"""
Script to generate Figure 5 of the paper.

Giovanni Picogna, 21.02.2023
"""
import os
import numpy as np
import matplotlib.pyplot as plt
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

fig, ax = plt.subplots(3, 3, figsize=[5., 5.])

path = "../data/"

fnames = ["image085", "image3"]
gap = [10, 20, 30]
dpc = 140.
au = 1.496e13
size = [100., 120., 140.]
beam_fwhm = [[0.034, 0.031], [0.067, 0.056]]
npix = 2048

linecolors = ["sandybrown", "saddlebrown"]
linestyle = ["-", "-."]

os.chdir(path)

for i in range(2):

    for j in range(3):

        im = image.readImage(fname=fnames[i]+str(gap[j])+'au.out')
        data = im.image[:, ::-1, 0].T
        clipmax = np.log10(data.max())
        clipmin = np.log10(data[data > 0.].min())
        data = np.log10(data.clip(1e-90))
        data = data.clip(clipmin, clipmax)
        # convolve image with beam
        im_conv = im.imConv(dpc=dpc, fwhm=beam_fwhm[i], pa=30.)
        data_conv = im_conv.image[:, ::-1, 0].T
        clipmax_conv = np.log10(data_conv.max())
        clipmin_conv = np.log10(data_conv[data_conv > 0.].min())
        data_conv = np.log10(data_conv.clip(1e-90))
        data_conv = data_conv.clip(clipmin_conv, clipmax_conv)

        # Convert to Jy/pixel
        data += np.log10(im_conv.sizepix_x * im_conv.sizepix_y /
                         (dpc * const.pc.value)**2. * 1e23)
        data_conv += np.log10(im_conv.sizepix_x * im_conv.sizepix_y /
                              (dpc * const.pc.value)**2. * 1e23)
        # Convert to Jy/au2
        data += np.log10(npix/size[j]*npix/size[j])
        data_conv += np.log10(npix/size[j]*npix/size[j])
        # Convert to Jy/arcsec2
        data += np.log10(dpc*dpc)
        data_conv += np.log10(dpc*dpc)

        x = im.x / au
        y = im.y / au

        img = ax[i][j].pcolormesh(x, y, data_conv, cmap='inferno',
                                  vmin=-1, vmax=5)

        if j == 0:
            ax[i][j].set_ylabel('Y [au]')
            ax[2][j].set_ylabel('$\\log(I)$ [Jy/arcsec$^2$]')
            ax[2][j].yaxis.set_label_coords(-.1, .5)
        else:
            ax[i][j].set_yticklabels([])
            ax[2][j].set_yticklabels([])

        if i == 0:
            ax[i][j].set_title(str(gap[j])+' au')
        else:
            ax[i][j].set_xlabel('X [au]')

        ax[2][j].plot(x, data[1024, :], color=linecolors[i], ls='--')
        ax[2][j].plot(x, data_conv[1024, :], color=linecolors[i], ls='-')

        if j == 0:
            ax[2][j].set_xlim(3., 55.)
        elif j == 1:
            ax[2][j].set_xlim(8., 65.)
        else:
            ax[2][j].set_xlim(13., 75.)

        ax[2][j].set_ylim(-0.75, 5.)
        ax[2][j].set_xlabel('X [au]')

        ax[i][j].set_xlim(-75., 75.)
        ax[i][j].set_ylim(-75., 75.)

fig.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.38, 0.03, 0.58])
fig.colorbar(img, cax=cbar_ax, label='$\\log(I)$ [Jy/arcsec$^2$]')

plt.savefig(fname='Fig5.png', dpi=400)
