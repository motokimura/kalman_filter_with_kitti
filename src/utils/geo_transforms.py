# references:
# [1] https://www.enri.go.jp/~fks442/K_MUSEN/1st/1st060428rev2.pdf

import numpy as np


# constant parameters defined in [1]
_a = 6378137.
_f = 1. / 298.257223563
_b = (1. - _f) * _a
_e = np.sqrt(_a ** 2. - _b ** 2.) / _a
_e_prime = np.sqrt(_a ** 2. - _b ** 2.) / _b


def Rx(theta):
    """rotation matrix around x-axis
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])


def Ry(theta):
    """rotation matrix around y-axis
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])


def Rz(theta):
    """rotation matrix around z-axis
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])


def lla_to_ecef(points_lla):
    """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
    N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
    """
    lon = np.radians(points_lla[0])  # [N,]
    lat = np.radians(points_lla[1])  # [N,]
    alt = points_lla[2]  # [N,]

    N = _a / np.sqrt(1. - (_e * np.sin(lat)) ** 2.)  # [N,]
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1. - _e ** 2.) + alt) * np.sin(lat)

    points_ecef = np.stack([x, y, z], axis=0)  # [3, N]
    return points_ecef


def ecef_to_enu(points_ecef, ref_lla):
    """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
    N x [x, y, z] coords measured in a local East-North-Up frame.
    """
    lon = np.radians(ref_lla[0])
    lat = np.radians(ref_lla[1])
    alt = ref_lla[2]

    ref_ecef = lla_to_ecef(ref_lla)  # [3,]

    relative = points_ecef - ref_ecef[:, np.newaxis]  # [3, N]

    R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
    points_enu = R @ relative  # [3, N]
    return points_enu


def lla_to_enu(points_lla, ref_lla):
    """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
    N x [x, y, z] coords measured in a local East-North-Up frame.
    """
    points_ecef = lla_to_ecef(points_lla)
    points_enu = ecef_to_enu(points_ecef, ref_lla)
    return points_enu


def enu_to_ecef(points_enu, ref_lla):
    """transform N x [x, y, z] coords measured in a local East-North-Up frame into
    N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
    """
    # inverse transformation of `ecef_to_enu`

    lon = np.radians(ref_lla[0])
    lat = np.radians(ref_lla[1])
    alt = ref_lla[2]

    ref_ecef = lla_to_ecef(ref_lla)  # [3,]

    R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
    R = R.T  # inverse rotation
    relative = R @ points_enu  # [3, N]

    points_ecef = ref_ecef[:, np.newaxis] + relative  # [3, N]
    return points_ecef


def ecef_to_lla(points_ecef):
    """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
    N x [longitude(deg), latitude(deg), altitude(m)] coords.
    """
    # approximate inverse transformation of `lla_to_ecef`
    
    x = points_ecef[0]  # [N,]
    y = points_ecef[1]  # [N,]
    z = points_ecef[2]  # [N,]

    p = np.sqrt(x ** 2. + y ** 2.)  # [N,]
    theta = np.arctan(z * _a / (p * _b))  # [N,]

    lon = np.arctan(y / x)  # [N,]
    lat = np.arctan(
        (z + (_e_prime ** 2.) * _b * (np.sin(theta) ** 3.)) / \
        (p - (_e ** 2.) * _a * (np.cos(theta)) ** 3.)
    )  # [N,]
    N = _a / np.sqrt(1. - (_e * np.sin(lat)) ** 2.)  # [N,]
    alt = p / np.cos(lat) - N  # [N,]

    lon = np.degrees(lon)
    lat = np.degrees(lat)

    points_lla = np.stack([lon, lat, alt], axis=0)  # [3, N]
    return points_lla


def enu_to_lla(points_enu, ref_lla):
    """transform N x [x, y, z] coords measured in a local East-North-Up frame into
    N x [longitude(deg), latitude(deg), altitude(m)] coords.
    """
    points_ecef = enu_to_ecef(points_enu, ref_lla)
    points_lla = ecef_to_lla(points_ecef)
    return points_lla

def get_rigid_transformation(calib_path):
    with open(calib_path, 'r') as f:
        calib = f.readlines()

    R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
    t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]

    T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
    
    return T
