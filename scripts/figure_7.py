"""
Script to generate Figure 7 of the paper.

Giovanni Picogna, 13.04.2023
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
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

fig, ax = plt.subplots(1, 1, figsize=[8.5, 8.5])

path = "../data/"

sub_paths = ["10au/test_highres/", "20au/test_highres/", "30au/test_highres/"]
fnames = ["image085", "image3"]
gap = [10, 20, 30]
spec_index = np.zeros((3, 2048))
styles = ["-", "-.", ":"]
labels = ["10 au", "20 au", "30 au"]

beam_fwhm = [0.067, 0.056]
shifts = [11., 23., 36.]
dpc = 140.
au = 1.496e13
idx = [1036, 1192, 1260]
spec_den = np.log10(352.71) - np.log10(99.93)

os.chdir(path)

for i in range(3):
    for j in range(2):
        im = image.readImage(fname=fnames[j]+str(gap[i])+'au.out')
        # convolve image with beam
        im_conv = im.imConv(dpc=dpc, fwhm=beam_fwhm, pa=30.)
        data_conv = im_conv.image[:, ::-1, 0].T
        x = (im_conv.x*u.cm).to(u.AU)

        clipmax_conv = np.log10(data_conv.max())
        clipmin_conv = np.log10(data_conv[data_conv > 0.].min())
        data_conv = np.log10(data_conv.clip(1e-90))
        data_conv = data_conv.clip(clipmin_conv, clipmax_conv)

        if j == 0:
            spec_index[i] += data_conv[1024, :]
        else:
            spec_index[i] -= data_conv[1024, :]

    spec_index[i] /= spec_den
    ax.plot(x[idx[i]:2000].value-shifts[i], spec_index[i, idx[i]:2000],
            label=labels[i], color='tab:orange', ls=styles[i])

    ax.set_xlim(-5., 10.)
    ax.set_ylim(1.5, 4.)

Nazari_int = np.loadtxt('Nazari_intermediate.dat')
Nazari_fast = np.loadtxt('Nazari_fast.dat')

ax.plot((Nazari_int[:, 0]*au*dpc*u.cm).to(u.AU).value-13.5-36.,
        Nazari_int[:, 1], label='30 M$_\\oplus$ int.', linestyle='--',
        color='tab:grey')
ax.plot((Nazari_fast[:, 0]*au*dpc*u.cm).to(u.AU).value-17.-36.,
        Nazari_fast[:, 1], label='30 M$_\\oplus$ fast', linestyle='-',
        color='tab:grey')

ax.vlines(0., 1.5, 5., color='black', linestyles='dashed')
ax.set_xlabel("R [au]")
ax.set_ylabel(r"$\alpha_{73}$")
plt.legend(ncol=2, loc="upper right")

plt.savefig(fname='Fig7.png', format='png', dpi=400)
