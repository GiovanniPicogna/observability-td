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

plt.rc('font', size=18.)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=18.)
plt.rc('ytick', labelsize=18.)
plt.rc('axes', labelsize=18.)
plt.rc('axes', linewidth=0.5)

plt.rcParams["errorbar.capsize"]

rscale = 10.0 * u.AU
mscale = 0.7 * u.solMass
vscale = np.sqrt(const.G*mscale/rscale)/2.0/np.pi
rhoscale = mscale/rscale**3
pscale = rhoscale*vscale**2
particle_sizes = 10

fig, ax = plt.subplots(4, 3, sharey=True, sharex=False, squeeze=True,
                       figsize=(18, 14.5))

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

        ax[size][i].set_ylim(0., .5)
        if i == 0:
            ax[size][i].set_xlim(8, 20)
        elif i == 1:
            ax[size][i].set_xlim(17, 35)
        else:
            ax[size][i].set_xlim(27, 50)
        ax[size][i].set_title(str(sizes[i])+' au')
        data = np.loadtxt('stream'+str(sizes[i])+'.dat')

        D = (get_field('data' + str(sizes[i]) + '.dbl.h5', steps_hydro[i],
                       'rho')[0] * rhoscale).to(u.g/u.cm**3)
        P = (get_field('data' + str(sizes[i]) + '.dbl.h5', steps_hydro[i],
                       'prs')[0] * pscale).to(u.barye)
        
        D = D.value
        P = P.value
        Cd = get_field('data' + str(sizes[i]) + '.dbl.h5', steps_hydro[i],
                       'cd')[0]
        vals = [-1, 2.e22]

        if i == 0:
            vals_pres = [-4.5, -4, -3.5, -3.]
            manual_locations = [(14.5, 0.3), (14., 0.2), (12.5, 0.14), (11.5, 0.02)]
        elif i == 1:
            vals_pres = [-5., -4.5, -4.25, -4.]
            manual_locations = [(27, 0.32), (27., 0.24), (27, 0.17), (25, 0.03)]
        else:
            vals_pres = [-5.25, -5, -4.75, -4.6]
            manual_locations = [(38, 0.3), (38, 0.2), (38., 0.1), (37, 0.02)]

        im = ax[size][i].contour(Rt, -Tht+np.pi/2., np.log10(P), vals_pres,
                                 linestyles='dashed',
                                 colors='gray')

        ax[size][i].clabel(im, inline=1, manual=manual_locations, fontsize=18)

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
                                     np.log10(value), vmax=.5, cmap='turbo')

        for k in range(1, 13):
            dat = data[data[:, 0] == k]
            contours = ax[size][i].contour(Rt, -Tht+np.pi/2., Cd, vals,
                                           linestyles='dashed', colors='red')

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

cbar2 = fig.colorbar(dim, ax=ax.ravel().tolist(), orientation='vertical',
                     label=labelplot2, location='left', aspect=30)
plt.savefig('Fig3.png', dpi=400)
