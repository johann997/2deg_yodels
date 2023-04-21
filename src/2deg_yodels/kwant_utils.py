import kwant as kw
from scipy.interpolate import interp2d
import numpy as np

from utils import default_fig, reduce_multiple


# constants
e = 1.602e-19  # electron charge (C)
hb = 6.626e-34 / 2 / np.pi  # Dirac constant (Js)
h = 6.626e-34  # Planck constant (Js)
nm = 1e-9  # nanometres
um_to_nm = 1e-6 / nm  # convert micrometres to nanometres
ms = 0.067 * 9.109e-31  # an effective mass of electrons in 2DEG

T = (
    hb * hb / 2 / nm / nm / ms / e
)  # constant to convert from eV (output of nn++) to kwant energy unit


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


def get_xyz_from_discretised_gates(gates, data_type="potential"):
    """
    Return x and y array and z array which is the sum of all z_data stored in discretised_gates
    under the key data_type
    """

    x_data = gates["x_axis"]
    y_data = gates["y_axis"]
    z_data = np.zeros((np.shape(y_data)[0], np.shape(x_data)[0]))

    index = -1
    for key, val in gates.items():
        if "val_" in key:
            index += 1
            gate_val = gates[key]["gate_val"]
            if data_type == "potential":
                z_data += gates[key][data_type] * gate_val
            else:
                z_data += gates[key][data_type]

    return x_data, y_data, z_data


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
        return 4 * t - -interpolated_potential(x, y)[0] / T

    def hopping(*args):
        "set the onsite terms"
        return -t

    x_lattice_spacing = np.arange(minx // a + 1, maxx // a, 1)  # CHECK: spacing 1 or a
    y_lattice_spacing = np.arange(miny // a + 1, maxy // a, 1)
    #     x_coordinate_spacing = np.linspace(minx, maxx, len(x_lattice_spacing))
    #     y_coordinate_spacing = np.linspace(miny, maxy, len(y_lattice_spacing))
    dist = dist * um_to_nm // a

    syst = kw.Builder()

    ##### setting up the bulk system #####
    for x in x_lattice_spacing:
        for y in y_lattice_spacing:
            site = lat(x, y)
            syst[site] = onsite

    syst[lat.neighbors()] = hopping

    ##### setting up leads #####
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

        x0, y0 = lead_coords[0], lead_coords[1]

        sym_lead = kw.TranslationalSymmetry((a * x_periodicity, a * y_periodicity))
        lead = kw.Builder(sym_lead)

        x0, y0 = lead_coords[0], lead_coords[1]  # lattice coordinate of starting lead

        ##### not using lead_shape #####
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

            lead[(lat(x, y) for x in x_lead_lattice_spacing)] = onsite
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

            lead[(lat(x, y) for y in y_lead_lattice_spacing)] = onsite
            lead[lat.neighbors()] = hopping

        #         kw.plot(lead)
        syst.attach_lead(lead)

    return syst


def get_interpolated_potential(discretised_gates):
    x, y, potential = get_xyz_from_discretised_gates(
        discretised_gates, data_type="potential"
    )
    x *= um_to_nm  # put in units of nm
    y *= um_to_nm
    interpolated_potential = interp2d(x, y, potential)

    return interpolated_potential


def make_kwant_system(discretised_gates, lead_coords, minx, maxx, miny, maxy, numpts):
    # x, y, potential = get_xyz_from_discretised_gates(
    #     discretised_gates, data_type="potential"
    # )
    # x *= um_to_nm  # put in units of nm
    # y *= um_to_nm
    interpolated_potential = get_interpolated_potential(discretised_gates)

    # a = 1  # lattice constant of the tight-binding system (nm)
    a = int(get_lattice_constant(minx, maxx, miny, maxy, numpts=numpts) * um_to_nm)

    # Interpolating the extracted electric potential to fit the lattice passed to Kwant
    # It is used by some of the functions defined before - the first set
    # interpolated_potential = interp2d(x, y, potential)

    ##### Creating a square lattice #####
    lat = kw.lattice.square(a)

    ##### Defining the tight binding model #####
    lead_coords *= um_to_nm

    qpc = make_system(
        interpolated_potential,
        lat,
        minx * um_to_nm,
        maxx * um_to_nm,
        miny * um_to_nm,
        maxy * um_to_nm,
        a=a,
        t=1 / a**2,
        T=T,
        dist=0.1,
    )

    return qpc


def plot_kwant_system(discretised_gates, lead_coords, minx, maxx, miny, maxy, numpts):
    qpc = make_kwant_system(
        discretised_gates, lead_coords, minx, maxx, miny, maxy, numpts
    )

    ###### A plot of the system #####
    return kw.plot(qpc)


def plot_kwant_potential(
    discretised_gates, lead_coords, minx, maxx, miny, maxy, numpts
):
    qpc = make_kwant_system(
        discretised_gates, lead_coords, minx, maxx, miny, maxy, numpts
    )
    interpolated_potential = get_interpolated_potential(discretised_gates)

    def qpc_potential(site):  # potential in the scattering region
        x, y = site.pos

        return -interpolated_potential(x, y)[0] / T

    return kw.plotter.map(
        qpc,
        lambda site: qpc_potential(
            site,
        ),
    )


# ##### Electric potential #####
# kw.plotter.map(
#     qpc,
#     lambda s: qpc_potential(
#         s,
#     ),
# )


# ##### Finalise system #####
# fqpc = qpc.finalized()

# # ##### Electronic band strucure of the first lead #####
# # kw.plotter.bands(fqpc.leads[0]);