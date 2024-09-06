"""
this file simply demo a path planning algorithm which based on Frenet Coordinate
the result are showing, base on the global path, we can planning the 
local path make sure it requires those constrains:

1) no more than the max speed, the max acceleration
2) does not collision the obstacles
3) minmize the Jerk, make the path time cost lowest.

"""
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
from cubic_spline import Spline2D
from polynomials import QuarticPolynomial, QuinticPolynomial
from collision_check import collision_check_single, Rectangle
from utils import calc_collision_cost, FrenetPath, calc_global_paths

# Parameter
MAX_SPEED = 20  # maximum speed [m/s]
MAX_ACCEL = 1.5  # maximum acceleration [m/ss]
MAX_DEACC = -2.5
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
TARGET_SPEED = 16  # target speed [m/s]

# cost weights
KJ = 0.1
KT = 0.1
KD = 5.0
KLAT = 1.0
KLON = 2.0
ego_colli_w_lon, ego_colli_w_lat = [25, 20, 0.5], [4, 20, 2]


def calc_frenet_paths(s0, s_d0, s_dd0, d0, d_d0, d_dd0, d_sample_list, v_sample_list, T, dt):

    frenet_paths = []

    # generate path to each offset goal
    for di in d_sample_list:

        fp = FrenetPath(dt)
        lat_qp = QuinticPolynomial(d0, d_d0, d_dd0, di, 0.0, 0.0, T)

        fp.t = [t for t in np.arange(0.0, T+dt, dt)]
        fp.d = [lat_qp.calc_point(t) for t in fp.t]
        fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
        fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
        fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

        # Loongitudinal motion planning (Velocity keeping)
        for tv in v_sample_list:
            tfp = copy.deepcopy(fp)
            lon_qp = QuarticPolynomial(s0, s_d0, s_dd0, tv, 0.0, T)

            tfp.s = [lon_qp.calc_point(t) for t in fp.t]
            tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            frenet_paths.append(tfp)
    return frenet_paths


def check_collision(fp, obs_trajs, vehicle_width=1.8, vehicle_length=4.):
    for obs_traj in obs_trajs:
        # print(len(fp.s), len(fp.x), len(fp.y), len(fp.yaw))
        # print(len(obs_traj.s), len(obs_traj.x), len(obs_traj.y), len(obs_traj.yaw))
        for i in range(len(obs_traj.s)):
            # print(i)
            ego_rectg = Rectangle(fp.x[i], fp.y[i], vehicle_width, vehicle_length, fp.yaw[i])
            obs_rectg = Rectangle(obs_traj.x[i], obs_traj.y[i], vehicle_width, vehicle_length, obs_traj.yaw[i])
            if collision_check_single(ego_rectg, obs_rectg):
                # plt.plot(fp.x[i], fp.y[i])
                # plt.plot(obs_traj.x[i], obs_traj.y[i])
                # plt.show()
                return True
    return False


def check_paths(fplist, ob, print_check=0):
    """
    check path above max speed, max a, does collision or not
    """
    okind = []
    for i in range(len(fplist)):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            if print_check: print('max v')
            continue
        elif any([a > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            if print_check: print('max acc')
            continue
        elif any([a < MAX_DEACC for a in fplist[i].s_dd]):  # Max accel check
            if print_check: print('min acc')
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            if print_check: print('max cur')
            continue
        elif check_collision(fplist[i], ob):
            if print_check: print('collision')
            continue
        okind.append(i)
    # print(okind)
    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp, s0, s_d0, s_dd0, d0, d_d0, d_dd0, d_sample_list, v_sample_list, ob, T, dt):

    fplist = calc_frenet_paths(s0, s_d0, s_dd0, d0, d_d0, d_dd0, d_sample_list, v_sample_list, T, dt)

    for i in range(len(fplist)):
        fplist[i] = calc_global_paths(fplist[i], csp)
    fplist = check_paths(fplist, ob)
    # add cost
    for fp in fplist:
        Jp = sum(np.power(fp.d_ddd, 2))  # square of jerk
        Js = sum(np.power(fp.s_ddd, 2))  # square of jerk
        # square of diff from target speed
        fp.cd = KD * fp.d[-1] ** 2
        fp.cv = KD * (TARGET_SPEED - fp.s_d[-1]) ** 2
        fp.cf = KLAT * fp.cd + KLON * fp.cv

        # safety cost
        for obs_traj in ob:
            fp.safety_cost += calc_collision_cost(fp, obs_traj, ego_colli_w_lon, ego_colli_w_lat)
        fp.total_cost = fp.cf + fp.safety_cost
        # print(fp.cd, fp.cv, fp.safety_cost)
    # find minimum cost path
    mincost = float("inf")
    bestpath = None
    for fp in fplist:
        if mincost >= fp.total_cost:
            mincost = fp.total_cost
            bestpath = fp
    return bestpath


def generate_reference_traj(path):
    ref_traj = np.zeros((len(path.s), 6))
    for i in range(ref_traj.shape[0]):
        # print(ref_traj[i, :])
        ref_traj[i, :] = np.array([[path.x[i], path.y[i], path.yaw[i], path.s_d[i], path.d_d[i], 0]])
    return ref_traj


def generate_target_course(x, y):
    csp = Spline2D(x, y) 
    s = np.arange(0, csp.s[-1], 0.1)
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))
    return rx, ry, ryaw, rk, csp


