import copy
import kwant as kw
from scipy.interpolate import interp2d
import numpy as np

from utils import (
    default_fig, 
    reduce_multiple, 
    get_xyz_from_discretised_gates
    )

#####################
##### constants #####
#####################
electron_charge = 1.602e-19  # electron charge (C)
hbar = 1.05e-34 # hbar
nm = 1e-9  # nanometres
um_to_nm = 1e-6 / nm  # convert micrometres to nanometres
j_to_ev = 6.24e18 # convert J to eV
effective_mass = 0.067 * 9.109e-31  # an effective mass of electrons in 2DEG
T = (hbar**2 / (2 * effective_mass * (nm)**2)) * j_to_ev



def get_lattice_constant(minx, maxx, miny, maxy, numpts=100):
    """
    Calculate lattic constant so there are atleast numpts in the narrow direction.
    """
    x_range = np.abs(maxx - minx)
    y_range = np.abs(maxy - miny)

    if x_range <= y_range:
        a = x_range / numpts
    else:
        a = y_range / numpts

    return a


def on_perimeter(x, y, x_array, y_array):
    "Return True if x and y are in the perimeter"

    del_x = np.abs(x_array[1] - x_array[0]) / 2
    del_y = np.abs(y_array[1] - y_array[0]) / 2
    dist_horizontal = np.min(
        np.abs(x_array - x)
    )  # min distance between left or right side
    dist_vertical = np.min(
        np.abs(y_array - y)
    )  # min distance between top or bottom side

    if (
        dist_horizontal <= dist_vertical
        and np.abs(y - y_array[0]) <= del_y
        or np.abs(y - y_array[-1]) <= del_y
    ):
        return True, "y"
    elif (
        dist_vertical < dist_horizontal
        and np.abs(x - x_array[0]) <= del_x
        or np.abs(x - x_array[-1]) <= del_x
    ):
        return True, "x"
    return False, "none"


def lead_symmetry(x_array, y_array, x_val, y_val, dist=0.2):
    """
    Return lead periodicity based on lead symmetry
    """
    X, Y = np.meshgrid(x_array, y_array)
    num_x, num_y = 0, 0

    for i, x_row in enumerate(X):
        for j, x in enumerate(x_row):
            y = Y[i, j]

            r = np.sqrt((x - x_val) ** 2 + (y - y_val) ** 2)
            on_perimeter_bool, direction = on_perimeter(x, y, x_array, y_array)
            if on_perimeter_bool and r <= dist:
                if direction == "x":
                    num_x += 1
                else:
                    num_y += 1

    x_direction = 1
    if np.abs(x_val - x_array[0]) < np.abs(x_val - x_array[-1]):
        x_direction = -1
    y_direction = 1
    if np.abs(y_val - y_array[0]) < np.abs(y_val - y_array[-1]):
        y_direction = -1

    #     periodicity = reduce_multiple([num_x, num_y])

    if num_x >= num_y:
        return x_direction, 0
    else:
        return 0, y_direction


def get_closest_coordinates(coordinate, x_array, y_array):
    """
    Return x and y values from the array which are closest to the coordinates
    """
    x, y = coordinate[0], coordinate[1]
    x_close = x_array[np.argmin(np.abs(x_array - x))]
    y_close = y_array[np.argmin(np.abs(y_array - y))]

    return x_close, y_close


def make_system(
    interpolated_potential,
    lat,
    minx,
    maxx,
    miny,
    maxy,
    lead1_coords,
    lead2_coords,
    a=1,
    t=1,
    T=1,
    dist=0.2,
):  # define the scattering region
    """
    Set up bulk system and add leads given their coordinates in um
    """

    def onsite(site):
        "set the onsite terms"
        x, y = site.pos
        return 4 * t - interpolated_potential(x, y)[0] / T
    
    def onsite_lead(site):
        "set the onsite terms for the lead"
        return 4 * t 

    def hopping(*args):
        "set the onsite terms"
        return - t

    x_lattice_spacing = np.arange(minx // a + 1, maxx // a, 1)  # CHECK: spacing 1 or a
    y_lattice_spacing = np.arange(miny // a + 1, maxy // a, 1)
    #     x_coordinate_spacing = np.linspace(minx, maxx, len(x_lattice_spacing))
    #     y_coordinate_spacing = np.linspace(miny, maxy, len(y_lattice_spacing))
    dist = dist * um_to_nm // a

    syst = kw.Builder()

    ######################################
    ##### setting up the bulk system #####
    ######################################
    for x in x_lattice_spacing:
        for y in y_lattice_spacing:
            site = lat(x, y)
            syst[site] = onsite

    syst[lat.neighbors()] = hopping

    ############################
    ##### setting up leads #####
    ############################
    for lead_coords in [lead1_coords, lead2_coords]:
        # get lead coords in lattice space
        lead_coords = get_closest_coordinates(
            lead_coords // a, x_lattice_spacing, y_lattice_spacing
        )

        # determine periodicity of lead (i.e. top, bottom, left or right side)
        x_periodicity, y_periodicity = lead_symmetry(
            x_lattice_spacing,
            y_lattice_spacing,
            lead_coords[0],
            lead_coords[1],
            dist=dist,
        )

        sym_lead = kw.TranslationalSymmetry((a * x_periodicity, a * y_periodicity))
        lead = kw.Builder(sym_lead)

        x0, y0 = lead_coords[0], lead_coords[1]  # lattice coordinate of starting lead

        if x_periodicity == 0:
            ##### setting up a vertical lead system #####
            lead_indexs = np.where(
                np.abs(x_lattice_spacing - x0) <= dist
            )  # index of lattice points within dist
            x_lead_lattice_spacing = x_lattice_spacing[lead_indexs]
            y_lead_lattice = (
                y_lattice_spacing[0] if y_periodicity < 0 else y_lattice_spacing[-1]
            )
            y = y_lead_lattice

            lead[(lat(x, y) for x in x_lead_lattice_spacing)] = onsite_lead
            lead[lat.neighbors()] = hopping

        else:
            ###### setting up a horizontal lead system #####
            lead_indexs = np.where(
                np.abs(y_lattice_spacing - y0) <= dist
            )  # index of lattice points within dist
            y_lead_lattice_spacing = y_lattice_spacing[lead_indexs]
            x_lead_lattice = (
                x_lattice_spacing[0] if x_periodicity < 0 else x_lattice_spacing[-1]
            )
            x = x_lead_lattice

            lead[(lat(x, y) for y in y_lead_lattice_spacing)] = onsite_lead
            lead[lat.neighbors()] = hopping

        #         kw.plot(lead)
        syst.attach_lead(lead)

    return syst


def get_interpolated_potential(discretised_gates):
    """
    return interp2d object which is built from a 2d array of all the gates summed potential
    distances are calculated in nm space

    Args:
        discretised_gates (dict): 

    Returns:
        interp2d: 
    """

    discretised_gates_copy = copy.deepcopy(discretised_gates)
    x, y, potential = get_xyz_from_discretised_gates(
        discretised_gates_copy, data_type="potential"
    )
    x *= um_to_nm  # put in units of nm
    y *= um_to_nm
    interpolated_potential = interp2d(x, y, potential)

    return interpolated_potential


def make_kwant_system(discretised_gates, lead_coords, minx, maxx, miny, maxy, numpts, a=0):

    interpolated_potential = get_interpolated_potential(discretised_gates)

    # lattice constant of the tight-binding system (nm)
    if a == 0:
        a = int(get_lattice_constant(minx, maxx, miny, maxy, numpts=numpts) * um_to_nm)
    else:
        a = int(a)

    if T == 1:
        t  = (hbar**2 / (2 * effective_mass * (a*nm)**2)) * j_to_ev  # in units of Energy
    else:
        t = 1/a**2


    ##### Creating a square lattice #####
    lat = kw.lattice.square(a)

    ##### Defining the tight binding model #####
    qpc = make_system(
        interpolated_potential,
        lat,
        minx * um_to_nm,
        maxx * um_to_nm,
        miny * um_to_nm,
        maxy * um_to_nm,
        lead_coords[0] * um_to_nm,
        lead_coords[1] * um_to_nm,
        a=a,
        t=t,
        T=T,
        dist=0.1,
    )

    return qpc


def plot_kwant_system(qpc, ax=None):
    
    ###### A plot of the system #####
    return kw.plotter.plot(qpc, show=False, ax=ax)


def plot_kwant_potential(discretised_gates, qpc):

    interpolated_potential = get_interpolated_potential(discretised_gates)

    def qpc_potential(site):  # potential in the scattering region
        x, y = site.pos

        return - interpolated_potential(x, y)[0] / T

    return kw.plotter.map(
        qpc,
        lambda site: qpc_potential(
            site,
        ),
    )


def _finalise_kwant_system(qpc):
    return qpc.finalized()


def plot_kwant_band_structure(qpc, lead_num=0, ax=None):

    fqpc = _finalise_kwant_system(qpc)

    return kw.plotter.bands(fqpc.leads[lead_num], show=False,  ax=ax)

    

def plot_kwant_density(qpc, energy, lead_num=0, ax=None):

    fqpc = _finalise_kwant_system(qpc)
    def density():
        wf = kw.wave_function(fqpc, energy) 
        return (abs(wf(lead_num))**2).sum(axis=0)
    
    d = density() 
    
    return kw.plotter.map(fqpc, d, show=False,  ax=ax)


def plot_kwant_info(qpc, energy, info_type='bands', lead_num=0, ax=None):

    fqpc = _finalise_kwant_system(qpc)

    if info_type == 'bands':
        return kw.plotter.bands(fqpc.leads[lead_num])
    elif info_type == 'current':
        return kw.plotter.current(fqpc.leads[lead_num])
    elif info_type == 'density':
        return kw.plotter.density(fqpc.leads[lead_num])


def get_kwant_transmission(qpc, energy=0, lead_out=1, lead_in=0):
    """
    returns the transmission in units of 2e^2/h
    """
    fqpc = _finalise_kwant_system(qpc)
    smatrix = kw.smatrix(fqpc, energy, in_leads=[0])
    return smatrix.transmission(lead_out, lead_in)


