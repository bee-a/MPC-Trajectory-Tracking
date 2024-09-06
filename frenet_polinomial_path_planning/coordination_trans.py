import numpy as np
import math


def cartesian2frenet(xx, yx, vx, ax, yawx, kappax, ref_path):
    r_p = ref_path.csp
    r_p_x = ref_path.x
    r_p_y = ref_path.y
    r_p_yaw = ref_path.yaw
    r_p_curvature = ref_path.c

    index = len(r_p_x) - 1
    # s should be the same as csp.s
    s = np.arange(0, r_p.s[-1], 0.1)
    for i in range(len(r_p_x) - 2):
        dot_pro1 = (xx - r_p_x[i]) * (r_p_x[i + 1] - r_p_x[i]) + \
                   (yx - r_p_y[i]) * (r_p_y[i + 1] - r_p_y[i])
        dot_pro2 = (xx - r_p_x[i + 1]) * (r_p_x[i + 2] - r_p_x[i + 1]) + \
                   (yx - r_p_y[i + 1]) * (r_p_y[i + 2] - r_p_y[i + 1])
        if dot_pro1 * dot_pro2 <= 0:
            index = i + 1
            break
    s = s[index]
    xr = r_p_x[index]
    yr = r_p_y[index]
    yawr = r_p_yaw[index]
    kappar = r_p_curvature[index]
    absolute_d = math.sqrt((xx - xr) ** 2 + (yx - yr) ** 2)
    d = np.sign((yx - yr) * math.cos(yawr)
                - (xx - xr) * math.sin(yawr)) * absolute_d
    s_d = vx * math.cos(yawx - yawr) / (1 - kappar * d)
    d_d = (1 - kappar * d) * math.tan(yawx - yawr)
    one_minus_over_cos = (1-kappar*d) / math.cos(yawx-yawr)
    # since index always greater than 1, no worry about it will exceed the dimension
    kappar_d = (r_p_curvature[index]-r_p_curvature[index-1]) / (s[index]-s[index-1])
    kdkd = kappar_d*d+kappar*d_d
    kxkr = kappax*one_minus_over_cos-kappar
    s_dd = ax*math.cos(yawx - yawr)-s_d**2*(d_d*kxkr-kdkd)
    s_dd = s_dd/(1-kappar*d)
    d_dd = -kdkd*math.tan(yawx-yawr)+one_minus_over_cos/math.cos(yawx-yawr)*kxkr
    return s, d, s_d, d_d, s_dd, d_dd


def frenet2cartesian(s, s_d, s_dd, d, d_d, d_dd, ref_path):
    index = np.argmin(abs(np.asarray(ref_path.s-s)))
    xr, yr, yawr, kappar = ref_path.x[index], ref_path.y[index], ref_path.yaw[index], ref_path.c[index]
    r_p_curvature = ref_path.c
    xx = xr-d*math.sin(yawr)
    yx = yr+d*math.cos(yawr)
    yawx = math.atan(d_d/(1-kappar*d)) + yawr
    one_minus_over_cos = (1 - kappar * d) / math.cos(yawx - yawr)
    cos_over_one_minus = 1/one_minus_over_cos
    kappar_d = (r_p_curvature[index] - r_p_curvature[index - 1]) / (s[index] - s[index - 1])
    kdkd = kappar_d*d+kappar*d_d
    vx = s_d * one_minus_over_cos
    double_d_tan = d_dd+kdkd*math.tan(yawx-yawr)
    kappax = (double_d_tan*math.cos(yawx-yawr)*cos_over_one_minus+kappar)*cos_over_one_minus
    kxkr = kappax * one_minus_over_cos - kappar
    ax = s_dd * one_minus_over_cos+s_d**2/math.cos(yawx-yawr)*(d_d*kxkr-kdkd)

    return xx, yx, vx, ax, yawx, kappax