import math

import casadi as ca
import casadi.tools as ca_tools
import numpy as np
import time


class MPC:
    def __init__(self, a_max=1.5, a_min=-3, v_max=20, delta_max=0.6, delta_min=-0.6, omega_max=np.pi / 4.0,
                 dt=0.1, horizon_n=40):
        self.a_max = a_max
        self.a_min = a_min
        self.v_max = v_max
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.omega_max = omega_max
        self.dt = dt
        self.N = horizon_n

    # def shift_movement(self, cur_state, cur_control):
    #     f_value = self.f(cur_state, cur_control)
    #     cur_state = cur_state + self.dt*f_value
    #     return cur_state

    def solver_initialization(self):
        kf, kr, lf, lr, m, Iz = -128916, -85944, 1.06, 1.85, 1412, 1536.7
        Lk = lf * kf - lr * kr
        Ts = 0.1
        self.opti = ca.Opti()
        # control input
        self.opt_controls = self.opti.variable(self.N, 2)
        a = self.opt_controls[:, 0]
        delta = self.opt_controls[:, 1]
        # state
        self.opt_states = self.opti.variable(self.N + 1, 6)
        a = self.opt_controls[:, 0]
        delta = self.opt_controls[:, 1]
        vx = self.opt_states[:, 3]
        # self.f = lambda x_, u_: ca.vertcat(*[u_[0]*np.cos(x_[2]), u_[0]*np.sin(x_[2]), u_[1]])
        # self.f_np = lambda x_, u_: np.array([u_[0]*np.cos(x_[2]), u_[0]*np.sin(x_[2]), u_[1]])
        # self.f for solver
        self.f = lambda X, U: ca.vertcat(X[0] + Ts * (X[3] * np.cos(X[2]) - X[4] * np.sin(X[2])),
                                         X[1] + Ts * (X[4] * np.cos(X[2]) + X[3] * np.sin(X[2])),
                                         X[2] + Ts * X[5],
                                         X[3] + Ts * U[0],
                                         (m * X[3] * X[4] + Ts * Lk * X[5] - Ts * kf * U[1] * X[3] - Ts * m * X[
                                             3] ** 2 * X[5]) / (m * X[3] - Ts * (kf + kr)),
                                         (Iz * X[3] * X[5] + Ts * Lk * X[4] - Ts * lf * kf * U[1] * X[3]) / (
                                                     Iz * X[3] - Ts * (lf ** 2 * kf + lr ** 2 * kr)))
        # self.f_np for state update
        self.f_np = lambda X, U: np.array([X[0] + Ts * (X[3] * np.cos(X[2]) - X[4] * np.sin(X[2])),
                                           X[1] + Ts * (X[4] * np.cos(X[2]) + X[3] * np.sin(X[2])),
                                           X[2] + Ts * X[5],
                                           X[3] + Ts * U[0],
                                           (m * X[3] * X[4] + Ts * Lk * X[5] - Ts * kf * U[1] * X[3] - Ts * m * X[3] ** 2 * X[5]) / (m * X[3] - Ts * (kf + kr)),
                                           (Iz * X[3] * X[5] + Ts * Lk * X[4] - Ts * lf * kf * U[1] * X[3]) / (Iz * X[3] - Ts * (lf ** 2 * kf + lr ** 2 * kr))])
        self.opt_x0 = self.opti.parameter(6)  # init state
        # ref state [x, y, phi]
        self.opt_x_ref = self.opti.parameter(self.N + 1, 6)  # ref state

        self.opti.subject_to(self.opt_states[0, :] == self.opt_x0.T)
        for i in range(self.N):
            # x(k+i+1|k) = f_d(x(k+i|k), u(k+i|k))
            x_next = self.f(self.opt_states[i, :], self.opt_controls[i, :]).T
            self.opti.subject_to(self.opt_states[i + 1, :] == x_next)

        Q = np.diag([10, 15, 5, 6, 6, 0])
        R = np.array([[0.5, 0.0],
                      [0.0, 0.5]])

        obj = 0
        for i in range(self.N):
            # todo  x_ref changes with timestep t
            ref_state = self.opt_x_ref[i, :]
            obj = obj + ca.mtimes([(self.opt_states[i, :] - ref_state), Q, (self.opt_states[i, :] - ref_state).T]) \
                  + ca.mtimes([self.opt_controls[i, :], R, self.opt_controls[i, :].T])
        ref_state = self.opt_x_ref[self.N, :]
        obj = obj + ca.mtimes([(self.opt_states[self.N, :] - ref_state), Q, (self.opt_states[self.N, :] - ref_state).T])

        # optimization objective
        self.opti.minimize(obj)

        self.opti.subject_to(self.opti.bounded(self.a_min, a, self.a_max))
        self.opti.subject_to(self.opti.bounded(self.delta_min, delta, self.delta_max))
        self.opti.subject_to(self.opti.bounded(0, vx, self.v_max))

        opts_setting = {'ipopt.max_iter': 200, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def get_solution(self, cur_state, cur_control, ref_traj):
        print(ref_traj.shape)
        self.solver_initialization()
        self.opti.set_value(self.opt_x_ref, ref_traj)
        pred_state = np.zeros((self.N + 1, 6))
        pred_cur = np.zeros((self.N, 2))
        pred_cur[0] = cur_control
        self.opti.set_value(self.opt_x0, cur_state)
        self.opti.set_initial(self.opt_controls, pred_cur)
        self.opti.set_initial(self.opt_states, pred_state)

        solution = self.opti.solve()
        # print(solution.value(ca.hessian(self.opti.f+ca.dot(self.opti.lam_g, self.opti.g), self.opti.x)[0]))
        pred_control = solution.value(self.opt_controls)
        pred_state = solution.value(self.opt_states)

        print(solution.value(self.opti.lam_g))
        return pred_control, pred_state


    def mpc_nonliner(self, cur_state, cur_control, ref_traj, final_state):
        self.solver_initialization()
        self.opti.set_value(self.opt_x_ref, final_state)
        pred_state = np.zeros((self.N + 1, 3))
        pred_cur = np.zeros((self.N, 2))
        pred_cur[0] = cur_control
        tot_control = []
        mpciter = 0
        while np.linalg.norm(cur_state - final_state) > 1e-1 and mpciter < 100:
            self.opti.set_value(self.opt_x0, cur_state)
            self.opti.set_initial(self.opt_controls, pred_cur)  # (N, 2)
            self.opti.set_initial(self.opt_states, pred_state)  # (N+1, 3)
            sol = self.opti.solve()

            pred_control = sol.value(self.opt_controls)
            tot_control.append(pred_control[0, :])
            pred_state = sol.value(self.opt_states)

            f_value = self.f_np(cur_state, pred_control[0])
            cur_state = cur_state + self.dt * f_value
            print(cur_state, pred_control[0])
            # update variable
            # pred_state = np.concatenate((pred_state[1:], pred_state[-1:]))
            # pred_control = np.concatenate((pred_control[1:], pred_control[-1:]))
            mpciter = mpciter + 1



if __name__ == '__main__':
    controller = MPC()
    print(1)
    cur_state = np.array([-1., -1., 0.])
    cur_control = np.array([0., 0.])
    final_state = np.array([1.5, 1.5, 0.])
    controller.mpc_nonliner(cur_state=cur_state, cur_control=cur_control, final_state=final_state)
