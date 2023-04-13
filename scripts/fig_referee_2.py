"""
Script to generate Figure 3 of the paper.

Giovanni Picogna, 20.02.2023
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from lib import get_cell_coordinates, get_field, read_particles

import scienceplots

plt.style.use('science')

plt.rc('font', size=12.)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=12.)
plt.rc('ytick', labelsize=12.)
plt.rc('axes', labelsize=12.)
plt.rc('axes', linewidth=0.5)

plt.rcParams["errorbar.capsize"]

rscale = 10.0 * u.AU
mscale = 0.7 * u.solMass
vscale = np.sqrt(const.G*mscale/rscale)/2.0/np.pi
rhoscale = mscale/rscale**3
pscale = rhoscale*vscale**2
MU = 1.37125
particle_sizes = 10

fig, ax = plt.subplots(4, 3, sharey=True, sharex=False, squeeze=True,
                       figsize=(16, 10.5))

basepath = '../data/'
sizes = [10, 20, 30]
steps_hydro = [324, 406, 438]

os.chdir(basepath)

for size in range(4):
    for i in range(3):
        ax[size][i].clear()

        pdata = read_particles('part_data'+str(sizes[i])+'.dbl')

        xcell, ycell, zcell = (get_cell_coordinates('data' +
                               str(sizes[i]) + '.dbl.h5') *
                               rscale).to(u.cm)
        X = xcell.value
        Z = zcell.value

        Rt = np.sqrt(X**2+Z**2)
        Tht = np.arctan2(np.sqrt(X**2), Z)
        Rt = (Rt*u.cm).to(u.AU).value

        ax[size][i].set_ylim(0., .6)
        if i == 0:
            ax[size][i].set_xlim(6, 20)
        elif i == 1:
            ax[size][i].set_xlim(14, 35)
        else:
            ax[size][i].set_xlim(21, 50)
        ax[size][i].set_title(str(sizes[i])+' au')
        data = np.loadtxt('stream'+str(sizes[i])+'.dat')

        D = (get_field('data' + str(sizes[i]) + '.dbl.h5', steps_hydro[i],
                       'rho')[0] * rhoscale).to(u.g/u.cm**3)
        P = (get_field('data' + str(sizes[i]) + '.dbl.h5', steps_hydro[i],
                       'prs')[0] * pscale).to(u.barye)
        Tgas = ((P*MU*const.m_p)/(const.k_B*D)).to(u.K).value
        Cd = get_field('data' + str(sizes[i]) + '.dbl.h5', steps_hydro[i],
                       'cd')[0]
        vals = [-1, 2.e22]

        im = ax[size][i].pcolormesh(Rt, -Tht+np.pi/2., np.log10(P.value),
                                    vmin=-6.,vmax=-3.,cmap=plt.cm.jet)

        nanfilter = ~(np.isnan(pdata['pos_x'][size::particle_sizes]))
        r = (pdata['pos_x'][size::10][nanfilter] * rscale).to(u.AU).value
        th = pdata['pos_y'][size::10][nanfilter]
        x = r*np.sin(th)
        z = r*np.cos(th)

        hist = np.histogram2d(r, th, bins=100)
        value = hist[0].transpose()
        clipmax = np.log10(value.max())
        clipmin = np.log10(value[value > 0.].min())
        value = np.log10(value.clip(1e-90))
        value = np.clip(value, clipmin, clipmax)
        dim = ax[size][i].pcolormesh(hist[1], -hist[2]+np.pi/2.,
                                     np.log10(value), vmax=.5, cmap='inferno')

        for k in range(1, 13):
            dat = data[data[:, 0] == k]
            contours = ax[size][i].contour(Rt, -Tht+np.pi/2., Cd, vals,
                                           linestyles='dashed', colors='black')

        if size == 3:
            ax[size][i].set_xlabel('r [AU]')
        if i == 0:
            ax[size][i].set_ylabel(r'colatitude [rad]')
        if i == 2:
            ax[size][i].yaxis.set_label_position("right")
            if size == 0:
                ax[size][i].set_ylabel('0.01 cm')
            elif size == 1:
                ax[size][i].set_ylabel('0.1 cm')
            elif size == 2:
                ax[size][i].set_ylabel('1 cm')
            else:
                ax[size][i].set_ylabel('10 cm')

labelplot = '$\\log_{10}$(P [barye])'
labelplot2 = '$\\Sigma_d$ [norm.]'

plt.tight_layout()
cbar = fig.colorbar(im, ax=ax.ravel().tolist(), orientation='vertical',
                    label=labelplot, location='right', aspect=30)
cbar2 = fig.colorbar(dim, ax=ax.ravel().tolist(), orientation='vertical',
                     label=labelplot2, location='left', aspect=30)
plt.savefig('Fig_referee_2.png', dpi=400)
