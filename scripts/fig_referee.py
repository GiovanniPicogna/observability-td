"""
Script to generate Figure 1 of the paper.

Giovanni Picogna, 24.06.2022
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from lib import get_cell_coordinates, get_field
import scienceplots
from astropy import constants as const

plt.style.use('science')
plt.rc('font', size=16.)
plt.rc('text', usetex=True)

fig, ax = plt.subplots(1, 3, sharey=False, sharex=False, squeeze=True,
                       figsize=(16, 5))

streams = ['stream10.dat', 'stream20.dat', 'stream30.dat']
output = ['data10.dbl.h5', 'data20.dbl.h5', 'data30.dbl.h5']
steps = [324, 406, 438]
titles = ['10 au', '20 au', '30 au']

rscale = 10.0 * u.AU
mscale = 0.7 * u.solMass
rhoscale = mscale/rscale**3
vscale = np.sqrt(const.G*mscale/rscale)/(2.*np.pi)
pscale = rhoscale*vscale**2
MU = 1.37125

os.chdir("../data")

for i in range(3):
    ax[i].clear()
    #ax[i].set_aspect('equal')

    xcell, ycell, zcell = (get_cell_coordinates(output[i]) * rscale).to(u.cm)
    X = xcell
    Z = zcell
    ax[i].set_xlim(0, 80)
    ax[i].set_ylim(0, 80)
    ax[i].set_title(titles[i], fontsize=16.)
    data = np.loadtxt(streams[i])

    D = (get_field(output[i], steps[i], 'rho')[0] * rhoscale).to(u.g/u.cm**3)
    P = (get_field(output[i], steps[i], 'prs')[0] * pscale).to(u.barye)
    Tgas = ((P*MU*const.m_p)/(const.k_B*D)).to(u.K).value
    Cd = get_field(output[i], steps[i], 'cd')[0]
    vals = [-1, 2.e22]

    xplot = X.to(u.AU).value
    zplot = Z.to(u.AU).value
    value = np.log10(Tgas)

    im = ax[i].pcolormesh(xplot, zplot, value, vmin=1, vmax=4, cmap=plt.cm.jet)

    for k in range(1, 13):
        dat = data[data[:, 0] == k]
        ax[i].plot((dat[:, 1]*u.cm).to(u.AU).value,
                   (dat[:, 2]*u.cm).to(u.AU).value, 'r-')
        contours = ax[i].contour(xplot, zplot, Cd, vals, colors='black',
                                 linestyles='dashed')

    LABEL = r'$\log_{10}(\rho)$ [$10^{-24}$ g cm$^{-3}$]'
    ax[i].set_xlabel('X [AU]')
    if i == 0:
        ax[i].set_xlim(5,15)
        ax[i].set_ylim(0,10)
    elif i == 1:
        ax[i].set_xlim(15,25)
        ax[i].set_ylim(0,15)
    else:
        ax[i].set_xlim(22,32)
        ax[i].set_ylim(0,20)

ax[0].set_ylabel('Z [AU]')
plt.tight_layout(pad=0.1)
cbar = fig.colorbar(im, ax=ax.ravel().tolist(), orientation='vertical',
                    label=LABEL)
plt.savefig('Fig_referee.pdf', dpi=400)
