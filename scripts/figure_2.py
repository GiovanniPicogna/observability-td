"""
Script to generate Figure 2 of the paper.

Giovanni Picogna, 24.06.2022
"""
import radmc3dPy
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from lib import get_cell_coordinates, get_field
from matplotlib.ticker import ScalarFormatter

import scienceplots

plt.style.use('science')

plt.rc('font', size=18.)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=18.)
plt.rc('ytick', labelsize=18.)
plt.rc('axes', labelsize=18.)
plt.rc('axes', linewidth=0.5)

plt.rcParams["errorbar.capsize"]

labels = [r'10 au', r'20 au', r'30 au']
gapnames = ["10au", "20au", "30au"]
fnames = ["image085", "image3"]
linecolors = ["tab:orange", "tab:red", "tab:brown"]
linestyle = ["-", "--"]
output = ['data10.dbl.h5', 'data20.dbl.h5', 'data30.dbl.h5']
beam_fwhm = [[0.034, 0.031], [0.067, 0.056]]
gap = [10., 20., 30.]
dpc = 140.
au = 1.496e13
size = [100., 120., 140.]
npix = 2048

rscale = 10.0 * u.AU
mscale = 1. * u.solMass
rhoscale = mscale/rscale**3
vscale = np.sqrt(const.G*mscale/rscale)/(2.*np.pi)
pscale = rhoscale*vscale**2

steps = [324, 406, 438]

fig, ax = plt.subplots(nrows=5, figsize=[8., 12.])

GAMMA = 1.4
MU = 1.37125

os.chdir("../data")

for i in range(3):

    xcell, ycell, zcell = (get_cell_coordinates(output[i]) * rscale).to(u.cm)
    X = xcell
    Z = zcell

    D = (get_field(output[i], steps[i], 'rho')[0] * rhoscale).to(u.g/u.cm**3)
    P = (get_field(output[i], steps[i], 'prs')[0] * pscale).to(u.barye)

    Omega = np.sqrt(const.G*0.7*mscale/(X[-1, 0:-1])**3).to(1./u.s)
    H = (np.sqrt(GAMMA*P[-1, :-1]/D[-1, :-1]) /
         (Omega*X[-1, :-1])).to(u.dimensionless_unscaled)

    ax[0].plot(X[-1, :-1]*u.cm.to(u.AU), H, '-', color=linecolors[i],
               label=labels[i])

    ax[0].set_ylabel('H/R')
    ax[0].set_ylim(0.05, 2.)
    ax[0].set_yscale("log")
    # ax[0].get_xaxis().set_ticklabels([])
    ax[0].text(0.05, 0.95, '(a)', transform=ax[0].transAxes, va='top')

    # ---------------------------

    Tgas = ((P*MU*const.m_p)/(const.k_B*D)).to(u.K).value
    ax[1].plot(X[-1, :-1]*u.cm.to(u.AU), Tgas[-1, :-1], '-',
               color=linecolors[i], label=labels[i])
    ax[1].set_ylabel(r'T$_\mathrm{gas}$ [K]')
    ax[1].set_ylim(10, 2.e4)
    ax[1].set_yscale("log")
    # ax[1].get_xaxis().set_ticklabels([])
    ax[1].text(0.05, 0.95, '(b)', transform=ax[1].transAxes,
               va='top')

    # ----------------------------

    ax[2].plot(X[-1, :-1]*u.cm.to(u.AU), D[-1, :-1], '-',
               color=linecolors[i], label=labels[i])
    ax[2].set_ylabel(r'$\rho_\mathrm{gas}$ [g cm$^{-3}$]')
    ax[2].set_ylim(1.e-18, 1.e-12)
    ax[2].set_yscale("log")
    # ax[2].get_xaxis().set_ticklabels([])
    ax[2].text(0.05, 0.95, '(c)', transform=ax[2].transAxes,
               va='top')

    # ----------------------------

    dPdR = np.asarray(np.diff(P))/np.asarray(np.diff(X))
    for j in range(np.size(X[-1, :-2])):
        if (dPdR[-1, j]*dPdR[-1, j+1] < 0):
            print(i, 0.5*(X[-1, j]*u.cm.to(u.AU)+X[-1, j+1]*u.cm.to(u.AU)))
    ax[3].plot(X[-1, :-1]*u.cm.to(u.AU),
               dPdR[-1, :]/np.max(dPdR[-1, :]),
               '-', color=linecolors[i], label=labels[i])
    ax[3].set_ylabel('dP/dR [norm.]')
    ax[3].text(0.05, 0.95, '(d)', transform=ax[3].transAxes,
               va='top')

    # ----------------------------

    for j in range(2):
        im = radmc3dPy.image.readImage(fname=fnames[j]+gapnames[i]+'.out')
        data_conv = im.image[:, ::-1, 0].T

        clipmax_conv = np.log10(data_conv.max())
        clipmin_conv = np.log10(data_conv[data_conv > 0.].min())
        data_conv = np.log10(data_conv.clip(1e-90))
        data_conv = data_conv.clip(clipmin_conv, clipmax_conv)

        # Convert data to Jy/pixel
        data_conv += np.log10(im.sizepix_x * im.sizepix_y /
                              (dpc * const.pc.value)**2. * 1e23)

        x = im.x*u.cm.to(u.AU)
        ax[4].plot(x[1024:], data_conv[1024, 1024:],
                   color=linecolors[i], ls=linestyle[j])

    ax[4].set_ylim(-7.5, -1.5)
    ax[4].set_ylabel(r'$log(I)$ [Jy px$^{-1}$]')
    ax[4].text(0.05, 0.95, '(e)', transform=ax[4].transAxes, va='top')

    # -----------------------------

    aymin = np.zeros((5))
    aymax = np.zeros((5))
    aymin[0] = 0.05
    aymax[0] = 2.
    aymin[3] = -0.5
    aymax[3] = 1
    aymin[2] = 1.e-18
    aymax[2] = 1.e-12
    aymin[1] = 10
    aymax[1] = 30000
    aymin[4] = -7.5
    aymax[4] = -1.5

    # for l in range(5):
    #    #ax[l].vlines(7.984342575073242,aymin[l],aymax[l],color='tab:orange',ls=':')
    #    ax[l].vlines(8.271406292915344,aymin[l],aymax[l],color='tab:orange',ls=':')
    #    ax[l].vlines(11.366876363754272,aymin[l],aymax[l],color='tab:orange',ls=':')

    #    #ax[l].vlines(16.98272466659546,aymin[l],aymax[l],color='tab:red',ls=':')
    #    ax[l].vlines(17.245429754257202,aymin[l],aymax[l],color='tab:red',ls=':')
    #    ax[l].vlines(18.41,aymin[l],aymax[l],color='tab:red',ls=':')
    #    ax[l].vlines(18.88,aymin[l],aymax[l],color='tab:red',ls=':')
    #    ax[l].vlines(24.547505378723145,aymin[l],aymax[l],color='tab:red',ls=':')

    #    ax[l].vlines(26.75,aymin[l],aymax[l],color='tab:brown',ls=':')
    #    ax[l].vlines(27.35,aymin[l],aymax[l],color='tab:brown',ls=':')
    #    ax[l].vlines(35.98134994506836,aymin[l],aymax[l],color='tab:brown',ls=':')

    for j in range(5):
        ax[j].set_xlim((6, 60))
        ax[j].set_xscale("log")
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        # formatter.set_major_formatter(ScalarFormatter("%2.1f"))
        ax[j].xaxis.set_major_formatter(formatter)
        # ax[j].xaxis.set_major_formatter(ScalarFormatter("%2.1f"))
        if j == 4:
            ax[j].set_xlabel('R [au]')

plt.tight_layout()

plt.savefig('Fig2.pdf', dpi=400)
