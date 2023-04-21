"""
Script to generate Figure 6 of the paper.

Giovanni Picogna, 23.02.2023
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from astropy import constants as const
from astropy import units as u
from radmc3dPy import image

plt.style.use('science')

plt.rc('font', size=18.)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=18.)
plt.rc('ytick', labelsize=18.)
plt.rc('axes', labelsize=18.)
plt.rc('lines', linewidth=3.)

plt.rcParams["errorbar.capsize"]

fig, ax = plt.subplots(1, 2, figsize=[17., 8.5])

path = "../data/"

fnames = ["image085", "image3"]
gap = [10, 20, 30]
dpc = 150.
beam_fwhm = [[0.03, 0.03], [0.067, 0.056]]
shifts = [7.046, 0.59555, -4.048]
shifts = [0., 0., 0.]
linestyles = ["-", "--", "-."]
labels = ["10 au", "20 au", "30 au"]
colors = ["tab:red", "tab:orange", "tab:purple"]

os.chdir(path)

for i in range(1):
    for j in range(3):
        im = image.readImage(fname=fnames[i]+str(gap[j])+'au.out')
        # convolve image with beam
        im_conv = im.imConv(dpc=dpc, fwhm=beam_fwhm[i], pa=0.)
        data_conv = im_conv.image[:, ::-1, 0].T
        clipmax_conv = data_conv.max()
        clipmin_conv = data_conv.min()
        data_conv = data_conv.clip(clipmin_conv, clipmax_conv)

        pixel_area = (im_conv.sizepix_x * im_conv.sizepix_y) /\
            (dpc * const.pc.value)**2 * (180./np.pi*3600.)**2
        beam_area = im_conv.fwhm[0] * im_conv.fwhm[1] * np.pi /\
            4. / np.log(2.0)

        # Convert data to Jy/pixel
        data_conv *= im_conv.sizepix_x * im_conv.sizepix_y /\
            (dpc * const.pc.value)**2. * 1e23
        # Convert data to Jy/beam
        data_conv *= beam_area / pixel_area

        x = im_conv.x*u.cm.to(u.AU)
        grad_data = data_conv[1024, 1024:]
        ax[0].plot(x[1024:], 1.e3*grad_data, linestyle=linestyles[j],
                   color="sandybrown")
        # ax[0].set_yscale('log')
        ax[0].set_xlim(3, 70)
        # ax[0].set_ylim(-1.5,0.6)
        ax[0].set_xlabel("R [au]")
        ax[0].set_ylabel(r'I (mJy beam$^{-1}$)', labelpad=15)

        if j == 2:
            Ovelar_1Mj = np.loadtxt('Ovelar_1Mj.dat')
            Ovelar_5Mj = np.loadtxt('Ovelar_3Mj.dat')
            Ovelar_9Mj = np.loadtxt('Ovelar_5Mj.dat')

            ax[0].plot(Ovelar_1Mj[:, 0], 1.e3*Ovelar_1Mj[:, 1],
                       linestyle='-', color="saddlebrown")
            ax[0].plot(Ovelar_5Mj[:, 0], 1.e3*Ovelar_5Mj[:, 1],
                       linestyle='--', color="saddlebrown")
            ax[0].plot(Ovelar_9Mj[:, 0], 1.e3*Ovelar_9Mj[:, 1],
                       linestyle='-.', color="saddlebrown")

        grad_data = np.gradient(data_conv[1024, 1024:], x[1024:])
        ax[1].plot(x[1024:], grad_data*1e4, label=labels[j],
                   linestyle=linestyles[j], color="sandybrown")

        if j == 2:
            ax[1].plot(Ovelar_1Mj[:, 0], 1e4*np.gradient(Ovelar_1Mj[:, 1],
                                                     Ovelar_1Mj[:, 0]),
                       label='1 M$_J$ @ 20 au', linestyle='-',
                       color="saddlebrown")
            ax[1].plot(Ovelar_5Mj[:, 0], 1e4*np.gradient(Ovelar_5Mj[:, 1],
                                                     Ovelar_5Mj[:, 0]),
                       label='5 M$_J$ @ 20 au', linestyle='--',
                       color="saddlebrown")
            ax[1].plot(Ovelar_9Mj[:, 0], 1e4*np.gradient(Ovelar_9Mj[:, 1],
                                                     Ovelar_9Mj[:, 0]),
                       label='9 M$_J$ @ 20 au', linestyle='-.',
                       color="saddlebrown")

        ax[1].set_xlim(3, 70)
        # ax.set_ylim(0,-0.0005)
        ax[1].set_xlabel("R [au]")
        ax[1].yaxis.set_label_position("right")
        ax[1].set_ylabel(r'$dI/dR \ [\cdot 10^{-4}$]')
        ax[1].yaxis.tick_right()
        ax[1].legend(ncol=2)
fig.tight_layout()
plt.savefig(fname='Fig6.png', format='png', dpi=400)
