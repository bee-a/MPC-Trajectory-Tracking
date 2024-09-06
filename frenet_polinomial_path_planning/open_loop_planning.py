import numpy as np
import matplotlib.pyplot as plt
from path_planning import generate_target_course, frenet_optimal_planning, generate_reference_traj
from utils import obs_predict_traj


# initialization
w_x = [0, 10, 20.5, 30, 40.5, 50, 60, 80, 100, 120, 150, 180, 230, 260, 280, 300, 330, 360, 400]
w_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tx, ty, tyaw, tc, csp = generate_target_course(w_x, w_y)
horizon = 40
dt = 0.1
T = horizon*dt
# initial state, all value are under Frenet Coordinate
obs_traj = obs_predict_traj(20, 12, 0, 0, 0,0, csp, horizon, dt)
obs_trajs = [obs_traj]
s0, s_d0, s_dd0, d0, d_d0, d_dd0 = 5, 12, 0.5, 0, 0, 0
d_sample_list, v_sample_list = np.linspace(-3.5, 3.5, 5), np.linspace(s0-5, s0+5, 5)
show_animation = True
area = 60

for i in range(500):
    # print('v_sample_list', v_sample_list)
    path = frenet_optimal_planning(csp, s0, s_d0, s_dd0, d0, d_d0, d_dd0,
                                   d_sample_list, v_sample_list, obs_trajs, T, dt)
    ref_traj = generate_reference_traj(path)


    # update
    s0, s_d0, s_dd0, d0, d_d0, d_dd0 = path.s[1], path.s_d[1], path.s_dd[1], path.d[1], path.d_d[1], path.d_dd[1]
    obs_traj = obs_predict_traj(obs_traj.s[1], obs_traj.s_d[1], obs_traj.s_dd[1],
                                obs_traj.d[1], obs_traj.d_d[1], obs_traj.d_dd[1],
                                csp, horizon, dt)
    obs_trajs = [obs_traj]
    d_sample_list, v_sample_list = np.linspace(-3.5, 3.5, 5), np.linspace(s_d0-5, s_d0+5, 5)
    if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
        print('Arrived.')
        break
    if show_animation:
        plt.cla()
        plt.plot(tx, ty)
        for obs_traj in obs_trajs:
            plt.plot(obs_traj.x[1:], obs_traj.y[1:], "ob")
        plt.plot(path.x[1:], path.y[1:], "-or")
        plt.plot(path.x[1], path.y[1], "vc")
        plt.xlim(path.x[1] - area, path.x[1] + area)
        plt.ylim(path.y[1] - area, path.y[1] + area)
        plt.title("v[km/h]:" + str(s_d0)[0:4]+' lateral:'+ str(d0)[0:4])
        plt.grid(True)
        plt.pause(0.0001)
print('Done')
if show_animation:
    plt.grid(True)
    plt.pause(0.0001)
    plt.show()
