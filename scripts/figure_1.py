"""
Script to generate Figure 1 of the paper.

Giovanni Picogna, 24.06.2022
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from lib import get_cell_coordinates, get_field

plt.rc('font', family='serif', serif='Times', size=15.)
plt.rc('text', usetex=True)

fig, ax = plt.subplots(1, 3, sharey=True, sharex=False, squeeze=True,
                       figsize=(16, 5))

streams = ['stream10.dat', 'stream20.dat', 'stream30.dat']
output = ['data10.dbl.h5', 'data20.dbl.h5', 'data30.dbl.h5']
steps = [324, 406, 438]
titles = ['10 au', '20 au', '30 au']

rscale = 10.0 * u.AU
mscale = 0.7 * u.solMass
rhoscale = mscale/rscale**3

os.chdir("../data")

for i in range(3):
    ax[i].clear()
    ax[i].set_aspect('equal')

    xcell, ycell, zcell = (get_cell_coordinates(output[i]) * rscale).to(u.cm)
    X = xcell[0]
    Z = zcell[0]
    ax[i].set_xlim(0, 80)
    ax[i].set_ylim(0, 80)
    ax[i].set_title(titles[i])
    data = np.loadtxt(streams[i])

    D = get_field(output[i], steps[i], 'rho')[0]
    D *= rhoscale.to(u.g/u.cm**3).value
    Cd = get_field(output[i], steps[i], 'cd')[0]
    vals = [-1, 2.e22]

    xplot = X.to(u.AU).value
    zplot = Z.to(u.AU).value
    value = np.log10(D*1.e24)

    im = ax[i].pcolormesh(xplot, zplot, value, vmin=4, vmax=9, cmap=plt.cm.jet)

    for k in range(1, 13):
        dat = data[data[:, 0] == k]
        ax[i].plot((dat[:, 1]*u.cm).to(u.AU).value,
                   (dat[:, 2]*u.cm).to(u.AU).value, 'r-')
        contours = ax[i].contour(xplot, zplot, Cd, vals, colors='black',
                                 linestyles='dashed')

    LABEL = r'$\log_{10}(\rho)$ [$10^{-24}$ g cm$^{-3}$]'
    ax[i].set_xlabel('X [AU]')

ax[0].set_ylabel('Z [AU]')
plt.tight_layout(pad=0.1)
cbar = fig.colorbar(im, ax=ax.ravel().tolist(), orientation='vertical',
                    label=LABEL)
plt.savefig('Fig1.png', dpi=400)
