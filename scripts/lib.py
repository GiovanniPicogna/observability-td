"""
Libraries with the main functions to read PLUTO files.

Giovanni Picogna, 24.06.2022
"""
import os
import struct
import h5py as h
import numpy as np


def get_filenames():
    """List and sort all the HDF5 output files. Returns a list of filenames."""
    filenames = [x for x in os.listdir('./') if '.h5' in x]
    filenames.sort()
    return filenames


def get_cell_coordinates(filename=None):
    """Take in a number n, returns the square of n."""
    if not filename:
        filename = get_filenames()[0]
    output = h.File(filename, 'r')
    x_coords = output['cell_coords']['X'][0]
    y_coords = output['cell_coords']['Y'][0]
    z_coords = output['cell_coords']['Z'][0]
    x_coords = x_coords.astype('float')
    y_coords = y_coords.astype('float')
    z_coords = z_coords.astype('float')
    output.close()
    return x_coords, y_coords, z_coords


def get_field(filename, step, name):
    """Take in a number n, returns the square of n."""
    output = h.File(filename, 'r')
    field = (output["Timestep_"+str(step)+"/vars"][name][:])
    output.close()
    return field


def get_step_str(step):
    """Take in a number n, returns the square of n."""
    step_str = str(step)
    while len(step_str) < 4:
        nsstr = '0'+step_str

    return nsstr


def read_particles(file_name):
    """Take in a number n, returns the square of n."""
    with open(file_name, "rb") as file_data:
        fmt1 = "<"+"i"
        fmt2 = "<"+"d"

        nb1 = struct.calcsize(fmt1)
        nb2 = struct.calcsize(fmt2)

        number_particles = struct.unpack(fmt1, file_data.read(nb1))[0]

        particles_step = struct.unpack(fmt1, file_data.read(nb1))[0]
        particles_time = struct.unpack(fmt2, file_data.read(nb2))[0]

        dtype = np.dtype([
            ("pid", np.int32),
            ("pcell_x", np.int32),
            ("pcell_y", np.int32),
            ("pcell_z", np.int32),
            ("pos_x", np.float64),
            ("pos_y", np.float64),
            ("pos_z", np.float64),
            ("vel_x", np.float64),
            ("vel_y", np.float64),
            ("vel_z", np.float64),
            ("tstop", np.float64),
            ])

        particles = np.fromfile(file_data, dtype=dtype)

    sorter = particles['pid'].argsort()
    particles_temp = particles[sorter]

    particles = {}
    particles.update({'NOP': number_particles, 'time': particles_time,
                      'step': particles_step})
    particles['pid'] = particles_temp['pid']
    particles['pcell_x'] = particles_temp['pcell_x']
    particles['pcell_y'] = particles_temp['pcell_y']
    particles['pcell_z'] = particles_temp['pcell_z']
    particles['pos_x'] = particles_temp['pos_x']
    particles['pos_y'] = particles_temp['pos_y']
    particles['pos_z'] = particles_temp['pos_z']
    particles['vel_x'] = particles_temp['vel_x']
    particles['vel_y'] = particles_temp['vel_y']
    particles['vel_z'] = particles_temp['vel_z']
    particles['tstop'] = particles_temp['tstop']

    return particles
