import math
import matplotlib.pyplot as plt
import numpy as np
from coordination_trans import frenet2cartesian


class FrenetPath:

    def __init__(self, dt):
        self.dt = dt
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0
        self.safety_cost = 0.
        self.total_cost = 0.

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        self.omiga = []

def calc_global_paths(fp, csp):
    # calc global positions
    for i in range(len(fp.s)):
        ix, iy = csp.calc_position(fp.s[i])
        if ix is None:
            print('can not find s', fp.s[i])
            break
        iyaw = csp.calc_yaw(fp.s[i])
        di = fp.d[i]
        fx = ix + di * math.cos(iyaw + math.pi / 2.0)
        fy = iy + di * math.sin(iyaw + math.pi / 2.0)
        fp.x.append(fx)
        fp.y.append(fy)

    # calc yaw and ds
    for i in range(len(fp.x) - 1):
        dx = fp.x[i + 1] - fp.x[i]
        dy = fp.y[i + 1] - fp.y[i]
        fp.yaw.append(math.atan2(dy, dx))
        fp.ds.append(math.sqrt(dx ** 2 + dy ** 2))

    fp.yaw.append(fp.yaw[-1])
    fp.ds.append(fp.ds[-1])

    # calc curvature
    for i in range(len(fp.yaw) - 1):
        fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
        fp.omiga.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.dt)
    fp.c.append(fp.c[-1])
    fp.omiga.append(fp.omiga[-1])

    return fp


def revise_sigmoid(dis, safe_distance, peak_value, cliff_fac):
    dis = dis - safe_distance
    return peak_value/(1 + math.exp(cliff_fac * dis))


def calc_collision_cost(ego_path, obs_path, ego_colli_w_lon, ego_colli_w_lat,):
    ego_total_ob_cost = 0.
    for i in range(len(ego_path.s)):
        lon_distance = abs(ego_path.s[i] - obs_path.s[i])
        lat_dis = abs(ego_path.d[i] - obs_path.d[i])
        ego_total_ob_cost += min(revise_sigmoid(lon_distance, ego_colli_w_lon[0], ego_colli_w_lon[1], ego_colli_w_lon[2]),
                                 revise_sigmoid(lat_dis, ego_colli_w_lat[0], ego_colli_w_lat[1], ego_colli_w_lat[2]))

    return ego_total_ob_cost


def obs_predict_traj(s0, s_d0, s_dd0, d0, d_d0, d_dd0, csp, horizon, dt):
    obs_traj = FrenetPath(dt)
    obs_traj.s.append(s0)
    obs_traj.s_d.append(s_d0)
    obs_traj.s_dd.append(s_dd0)
    obs_traj.d.append(d0)
    obs_traj.d_d.append(d_d0)
    obs_traj.d_dd.append(d_dd0)
    for i in range(horizon):
        obs_traj.s_dd.append(s_dd0)
        obs_traj.s_d.append(obs_traj.s_d[i]+obs_traj.s_dd[i]*dt)
        # print(obs_traj.s[i], obs_traj.s_d[i], obs_traj.s_d[i+1], dt)
        obs_traj.s.append(obs_traj.s[i]+0.5*(obs_traj.s_d[i]+obs_traj.s_d[i+1])*dt)
        obs_traj.d_dd.append(d_dd0)
        obs_traj.d_d.append(obs_traj.d_d[i]+obs_traj.d_dd[i]*dt)
        obs_traj.d.append(obs_traj.d[i]+0.5*(obs_traj.d_d[i]+obs_traj.d_d[i+1])*dt)

    obs_traj = calc_global_paths(obs_traj, csp)
    return obs_traj


class RefPath:
    def __init__(self, x, y, yaw, c, csp):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.c = c
        self.csp = csp


def dynamic_state2frenet_state(ref_path, dynamic_state, ax, kappax):
    px, py, phi, lat_v, long_v, omega = (dynamic_state[0], dynamic_state[1], dynamic_state[2],
                                         dynamic_state[3], dynamic_state[4], dynamic_state[5],)
    # todo v from ego frame to cartesian frame
    vx = math.sqrt(lat_v**2+long_v**2)
    r_p = ref_path.csp
    r_p_x = ref_path.x
    r_p_y = ref_path.y
    r_p_yaw = ref_path.yaw
    r_p_curvature = ref_path.c

    index = len(r_p_x) - 1
    # s should be the same as csp.s
    all_s = np.arange(0, r_p.s[-1], 0.1)
    for i in range(len(r_p_x) - 2):
        dot_pro1 = (px - r_p_x[i]) * (r_p_x[i + 1] - r_p_x[i]) + \
                   (py - r_p_y[i]) * (r_p_y[i + 1] - r_p_y[i])
        dot_pro2 = (px - r_p_x[i + 1]) * (r_p_x[i + 2] - r_p_x[i + 1]) + \
                   (py - r_p_y[i + 1]) * (r_p_y[i + 2] - r_p_y[i + 1])
        if dot_pro1 * dot_pro2 <= 0:
            index = i + 1
            break
    s = all_s[index]
    xr = r_p_x[index]
    yr = r_p_y[index]
    yawr = r_p_yaw[index]
    kappar = r_p_curvature[index]
    absolute_d = math.sqrt((px - xr) ** 2 + (py - yr) ** 2)
    d = np.sign((py - yr) * math.cos(yawr)
                - (px - xr) * math.sin(yawr)) * absolute_d
    s_d = vx * math.cos(phi - yawr) / (1 - kappar * d)
    d_d = (1 - kappar * d) * math.tan(phi - yawr)
    one_minus_over_cos = (1-kappar*d) / math.cos(phi-yawr)
    # since index always greater than 1, no worry about it will exceed the dimension
    kappar_d = (r_p_curvature[index]-r_p_curvature[index-1]) / (all_s[index]-all_s[index-1])
    kdkd = kappar_d*d+kappar*d_d
    kxkr = kappax*one_minus_over_cos-kappar
    s_dd = ax*math.cos(phi - yawr)-s_d**2*(d_d*kxkr-kdkd)
    s_dd = s_dd/(1-kappar*d)
    d_dd = -kdkd*math.tan(phi-yawr)+one_minus_over_cos/math.cos(phi-yawr)*kxkr
    return s, s_d, s_dd, d, d_d, d_dd




if __name__ == '__main__':
    dis = np.linspace(10, 0, 1000)
    cost = []
    for d in dis:
        cost.append(revise_sigmoid(d, 3, 10, 2))
    plt.plot(dis, cost)
    plt.show()
