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
from matplotlib import ticker

import scienceplots

plt.style.use('science')

plt.rc('font', size=18.)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=18.)
plt.rc('ytick', labelsize=18.)
plt.rc('axes', labelsize=18.)
plt.rc('lines', linewidth=3.)

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

    ax[0].set_ylabel('H/R', labelpad=15)
    ax[0].set_ylim(0.05, 2.)
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.1f}"))
    # ax[0].get_xaxis().set_ticklabels([])
    ax[0].text(0.05, 0.95, '(a)', transform=ax[0].transAxes, va='top')

    # ---------------------------

    Tgas = ((P*MU*const.m_p)/(const.k_B*D)).to(u.K).value
    ax[1].plot(X[-1, :-1]*u.cm.to(u.AU), np.log10(Tgas[-1, :-1]), '-',
               color=linecolors[i], label=labels[i])
    ax[1].set_ylabel(r'$\log_{10}(T_\mathrm{gas}$ [K])', labelpad=26)
    ax[1].set_ylim(1.2, 4)
    # ax[1].set_yscale("log")
    # ax[1].get_xaxis().set_ticklabels([])
    ax[1].text(0.05, 0.95, '(b)', transform=ax[1].transAxes,
               va='top')

    # ----------------------------

    ax[2].plot(X[-1, :-1]*u.cm.to(u.AU), np.log10(D[-1, :-1].value), '-',
               color=linecolors[i], label=labels[i])
    ax[2].set_ylabel(r'$\log_{10}(\rho_\mathrm{gas}$ [g cm$^{-3}$])')
    ax[2].set_ylim(-18, -12)
    #ax[2].set_yscale("log")
    # ax[2].get_xaxis().set_ticklabels([])
    ax[2].text(0.05, 0.95, '(c)', transform=ax[2].transAxes,
               va='top')

    # ----------------------------

    dPdR = np.asarray(np.diff(P))/np.asarray(np.diff(X))
#    for j in range(np.size(X[-1, :-2])):
#        if (dPdR[-1, j]*dPdR[-1, j+1] < 0):
#            print(i, 0.5*(X[-1, j]*u.cm.to(u.AU)+X[-1, j+1]*u.cm.to(u.AU)))
    ax[3].plot(X[-1, :-1]*u.cm.to(u.AU),
               dPdR[-1, :]/np.max(dPdR[-1, :]),
               '-', color=linecolors[i], label=labels[i])
    ax[3].set_ylabel('$dP/dR$ [norm.]', labelpad=0)
    ax[3].set_ylim(-0.9, 1.)
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
    ax[4].set_ylabel(r'$log_{10}(I)$ [Jy px$^{-1}$]', labelpad=11)
    ax[4].text(0.05, 0.95, '(e)', transform=ax[4].transAxes, va='top')

    # -----------------------------

    aymin = np.zeros((5))
    aymax = np.zeros((5))
    aymin[0] = 0.05
    aymax[0] = 2.
    aymin[3] = -1
    aymax[3] = 1.
    aymin[2] = -18
    aymax[2] = -12
    aymin[1] = 1.2
    aymax[1] = 4
    aymin[4] = -7.5
    aymax[4] = -1.5

    for l in range(5):
        ax[l].vlines(8.2, aymin[l], aymax[l], alpha=0.333, color='tab:orange', ls=':')
        ax[l].vlines(8.5, aymin[l], aymax[l], alpha=0.333, color='tab:orange', ls=':')
        ax[l].vlines(9.04, aymin[l], aymax[l], alpha=0.333, color='tab:orange', ls=':')
        ax[l].vlines(11.7, aymin[l], aymax[l], alpha=0.333, color='tab:orange', ls=':')

        ax[l].vlines(17.11, aymin[l], aymax[l], alpha=0.333, color='tab:red', ls=':')
        ax[l].vlines(18.48, aymin[l], aymax[l], alpha=0.333, color='tab:red', ls=':')
        ax[l].vlines(19.06, aymin[l], aymax[l], alpha=0.333, color='tab:red', ls=':')
        ax[l].vlines(22.37, aymin[l], aymax[l], alpha=0.333, color='tab:red', ls=':')
        ax[l].vlines(25.12, aymin[l], aymax[l], alpha=0.333, color='tab:red', ls=':')

        ax[l].vlines(27.01, aymin[l], aymax[l], alpha=0.333, color='tab:brown', ls=':')
        ax[l].vlines(27.77, aymin[l], aymax[l], alpha=0.333, color='tab:brown', ls=':')
        ax[l].vlines(36.23, aymin[l], aymax[l], alpha=0.333, color='tab:brown', ls=':')
        ax[l].vlines(38.32, aymin[l], aymax[l], alpha=0.333, color='tab:brown', ls=':')
        ax[l].vlines(39.41, aymin[l], aymax[l], alpha=0.333, color='tab:brown', ls=':')

    for j in range(5):
        ax[j].set_xlim((6, 60))
        ax[j].set_xscale("log")
        if j == 4:
            ax[j].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.0f}"))
            ax[j].xaxis.set_minor_formatter(ticker.StrMethodFormatter("{x:1.0f}"))
            ax[j].set_xlabel('R [au]')
        else:
            #ax[j].set_xticklabels([], minor=True)
            ax[j].set_xticklabels([])
            ax[j].set_xticklabels([], minor=True)

fig.tight_layout(pad=0.1)

fig.savefig('Fig2.png', format="png", dpi=400)
