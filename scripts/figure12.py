"""
Script to generate Figure 1-bis of the paper.

Giovanni Picogna, 24.06.2022
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from lib import get_cell_coordinates, get_field
import scienceplots

plt.style.use('science')
plt.rc('font', size=16.)
plt.rc('text', usetex=True)

fig, ax = plt.subplots(2, 3, sharey=True, sharex=False, squeeze=True,
                       figsize=(16, 10))

streams = ['stream10.dat', 'stream20.dat', 'stream30.dat']
output = ['data10.dbl.h5', 'data20.dbl.h5', 'data30.dbl.h5']
steps = [324, 406, 438]
titles = ['10 au', '20 au', '30 au']

rscale = 10.0 * u.AU
mscale = 0.7 * u.solMass
rhoscale = mscale/rscale**3
vscale = np.sqrt(const.G*mscale/rscale)/2.0/np.pi
mu = 1.37125

os.chdir("../data")

for i in range(3):

    xcell, ycell, zcell = (get_cell_coordinates(output[i]) * rscale).to(u.cm)
    X = xcell
    Z = zcell

    data = np.loadtxt(streams[i])

    D = get_field(output[i], steps[i], 'rho')[0]
    D = (D*rhoscale).to(u.g/u.cm**3)
    P = get_field(output[i], steps[i], 'prs')[0]
    P = (P*vscale**2*rhoscale).to(u.barye)
    T = ((P*mu*const.m_p)/(const.k_B*D)).to(u.K).value
    D = D.value
    Cd = get_field(output[i], steps[i], 'cd')[0]
    vals = [-1, 2.e22]

    xplot = X.to(u.AU).value
    zplot = Z.to(u.AU).value
    value = np.log10(D*1.e24)

    for j in range(2):
        ax[j][i].clear()
        ax[j][i].set_aspect('equal')
        ax[j][i].set_xlim(0, 80)
        ax[j][i].set_ylim(0, 80)
        ax[j][i].set_title(titles[i], fontsize=16.)
        if i == 0:
            ax[j][i].set_ylabel('Z [AU]')
        if j == 0:
            im = ax[j][i].pcolormesh(xplot, zplot, value, vmin=4, vmax=9, cmap=plt.cm.turbo)
            LABEL = r'$\\log_{10}(\rho)$ [$10^{-24}$ g cm$^{-3}$]'
            if i == 2:
                cbar = fig.colorbar(im, ax=ax[j][i], orientation='vertical',
                                    label=LABEL)
        else:
            value = np.log10(T)
            im = ax[j][i].pcolormesh(xplot, zplot, value, vmin=1, vmax=4, cmap=plt.cm.turbo)
            LABEL = r'$\\log_{10}(T)$ [K]'
            if i == 2:
                cbar = fig.colorbar(im, ax=ax[j][i], orientation='vertical',
                                    label=LABEL)
            ax[j][i].set_xlabel('X [AU]')

        for k in range(1, 13):
            dat = data[data[:, 0] == k]
            if j == 0:
                lc = 'r'
            else:
                lc = 'k'
            ax[j][i].plot((dat[:, 1]*u.cm).to(u.AU).value,
                    (dat[:, 2]*u.cm).to(u.AU).value, color=lc)
            contours = ax[j][i].contour(xplot, zplot, Cd, vals, colors='black',
                                        linestyles='dashed')

plt.tight_layout(pad=0.1)
plt.savefig('Fig12.png', dpi=400)
