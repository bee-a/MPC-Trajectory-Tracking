import math

import numpy as np
import matplotlib.pyplot as plt
from path_planning import generate_target_course, frenet_optimal_planning, generate_reference_traj, TARGET_SPEED
from mpc import MPC
from utils import obs_predict_traj, dynamic_state2frenet_state, RefPath


# initialization
w_x = [0, 10, 20.5, 30, 40.5, 50, 60, 80, 100, 120, 150, 180, 230, 260, 280, 300, 330, 360, 400]
w_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tx, ty, tyaw, tc, csp = generate_target_course(w_x, w_y)
ref_path = RefPath(tx, ty, tyaw, tc, csp)
horizon = 50
dt = 0.1
T = horizon*dt
# initial state, all value are under Frenet Coordinate
obs_traj = obs_predict_traj(20, 12, 0, 0, 0,0, csp, horizon, dt)
obs_trajs = [obs_traj]
s0, s_d0, s_dd0, d0, d_d0, d_dd0 = 10, 12, 0, 0, 0, 0
d_sample_list, v_sample_array = np.linspace(0, 3.5, 5).tolist(), np.linspace(s_d0 - 5, s_d0 + 5, 5)
v_sample = abs(v_sample_array - TARGET_SPEED)
if min(v_sample) < 0.5:
    v_sample_array[np.argmin(v_sample)] = TARGET_SPEED
    v_sample_list = v_sample_array.tolist()
else:
    v_sample_list = v_sample_array.tolist() + [TARGET_SPEED]


path = frenet_optimal_planning(csp, s0, s_d0, s_dd0, d0, d_d0, d_dd0,
                               d_sample_list, v_sample_list, obs_trajs, T, dt)
ref_traj = generate_reference_traj(path)
cur_state = np.array([path.x[0], path.y[0], path.yaw[0], path.s_d[0], path.d_d[0], 0.0])
cur_control = np.array([[0, 0]])
show_animation = True
area = 60
mpc_solver = MPC(horizon_n=horizon)
for i in range(500):
    # print(i)
    # print('v_sample_list', v_sample_list)

    pred_controls, pred_states = mpc_solver.get_solution(cur_state, cur_control, ref_traj)
    if not show_animation:
        plt.plot(path.t, pred_states[:, 1])
        plt.plot(path.t, path.y)
        plt.show()
    # print(pred_states)

    # update
    cur_state = mpc_solver.f_np(cur_state, pred_controls[0, :])
    cur_control = pred_controls[0, :]

    # print('control: ', pred_controls[0, :])
    long_acc = (cur_state[3]-pred_states[0, 3])/dt
    lat_acc = (cur_state[4]-pred_states[0, 4])/dt
    a = math.sqrt(long_acc**2+lat_acc**2)

    s0, s_d0, s_dd0, d0, d_d0, d_dd0 = path.s[1], path.s_d[1], path.s_dd[1], path.d[1], path.d_d[1], path.d_dd[1]
    # s0, s_d0, s_dd0, d0, d_d0, d_dd0 = cur_state[0], cur_state[]
    # print('planned s0, s_d0, s_dd0, d0, d_d0, d_dd0: ',
    #       path.s[1], path.s_d[1], path.s_dd[1], path.d[1], path.d_d[1], path.d_dd[1])
    # print('s0, s_d0, s_dd0, d0, d_d0, d_dd0: ', s0, s_d0, s_dd0, d0, d_d0, d_dd0)
    obs_traj = obs_predict_traj(obs_traj.s[1], obs_traj.s_d[1], obs_traj.s_dd[1],
                                obs_traj.d[1], obs_traj.d_d[1], obs_traj.d_dd[1],
                                csp, horizon, dt)
    obs_trajs = [obs_traj]
    d_sample_list, v_sample_array = np.linspace(0, 3.5, 5).tolist(), np.linspace(s_d0 - 5, s_d0 + 5, 5)
    v_sample = abs(v_sample_array - TARGET_SPEED)
    if min(v_sample) < 0.5:
        v_sample_array[np.argmin(v_sample)] = TARGET_SPEED
        v_sample_list = v_sample_array.tolist()
    else:
        v_sample_list = v_sample_array.tolist() + [TARGET_SPEED]
    path = frenet_optimal_planning(csp, s0, s_d0, s_dd0, d0, d_d0, d_dd0,
                                   d_sample_list, v_sample_list, obs_trajs, T, dt)
    ref_traj = generate_reference_traj(path)
    if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
        print('Arrived.')
        break
    if show_animation:
        plt.cla()
        plt.plot(tx, ty)
        for obs_traj in obs_trajs:
            plt.plot(obs_traj.x[1:], obs_traj.y[1:], "ob")
        plt.plot(pred_states[:, 0], pred_states[:, 1], "-or")
        plt.plot(pred_states[0, 0], pred_states[0, 1], "*k")
        plt.xlim(path.x[1] - 0.5*area, path.x[1] + 1.5*area)
        plt.ylim(path.y[1] - 0.5*area, path.y[1] + .5*area)
        plt.title("v[km/h]:" + str(s_d0)[0:4]+' lateral:'+ str(d0)[0:4])
        plt.grid(True)
        plt.pause(0.0001)
print('Done')
if show_animation:
    plt.grid(True)
    plt.pause(0.0001)
    plt.show()
