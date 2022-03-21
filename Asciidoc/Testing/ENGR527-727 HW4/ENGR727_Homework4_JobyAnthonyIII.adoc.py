# -*- coding:utf-8 -*-
from mako import runtime, filters, cache
UNDEFINED = runtime.UNDEFINED
STOP_RENDERING = runtime.STOP_RENDERING
__M_dict_builtin = dict
__M_locals_builtin = locals
_magic_number = 10
_modified_time = 1647874618.2761233
_enable_loop = True
_template_filename = 'C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Testing\\ENGR527-727 HW4\\ENGR727_Homework4_JobyAnthonyIII.adoc'
_template_uri = 'C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Testing\\ENGR527-727 HW4\\ENGR727_Homework4_JobyAnthonyIII.adoc'
_source_encoding = 'utf-8'
_exports = []



from engineering_notation import EngNumber as engr
from joby_m_anthony_iii import numerical_methods as nm
import numpy as np
import sympy as sp



A0_1, B0_1, C0_1, D0_1 = (0, 1), (1, 1), (1, 0), (0, 0) # m
A1_1, B1_1, C1_1, D1_1 = (-0.003, 0.0025), (-0.005, 0.0035), (-0.002, 0.001), (0, 0) # m
theta_1 = np.radians(30) # rad
sym_c_1 = [sp.Symbol("c1"), sp.Symbol("c2"), sp.Symbol("c3"), sp.Symbol("c4")]
sym_d_1 = [sp.Symbol("d1"), sp.Symbol("d2"), sp.Symbol("d3"), sp.Symbol("d4")]
u_1 = lambda c, x, y: c[0] + c[1]*x + c[2]*y + c[3]*x*y
v_1 = lambda d, x, y: d[0] + d[1]*x + d[2]*y + d[3]*x*y
sym_xy_1 = sym_x_1, sym_y_1 = sp.Symbol("x"), sp.Symbol("y")
sym_u_1 = sp.expand(u_1(sym_c_1, *sym_xy_1))
sym_v_1 = sp.expand(v_1(sym_d_1, *sym_xy_1))

Auv_1 = np.array([\
        # at A
    [1, A0_1[0], A0_1[1], A0_1[0]*A0_1[1]], \
        # at B
    [1, B0_1[0], B0_1[1], B0_1[0]*B0_1[1]], \
        # at C
    [1, C0_1[0], C0_1[1], C0_1[0]*C0_1[1]], \
        # at D
    [1, D0_1[0], D0_1[1], D0_1[0]*D0_1[1]], \
    ])

bu_1 = np.array([A1_1[0], B1_1[0], C1_1[0], D1_1[0]]).reshape((len(Auv_1), 1))
x0u_1 = np.ones_like(bu_1)
c_1 = np.linalg.lstsq(Auv_1, bu_1)[0].reshape(1, len(x0u_1))[0]

bv_1 = np.array([A1_1[1], B1_1[1], C1_1[1], D1_1[1]]).reshape((len(Auv_1), 1))
x0v_1 = np.ones_like(bv_1)
d_1 = np.linalg.lstsq(Auv_1, bv_1)[0].reshape(1, len(x0v_1))[0]

# build string function from constants
f_1 = lambda sym_function, sym_variables, constants: sp.lambdify(sym_xy_1, sp.lambdify(sym_variables, sym_function)(*constants))
u_1 = lambda point: f_1(sym_u_1, (sym_c_1), c_1)(*point)
sym_u_1 = sp.expand(u_1(sym_xy_1))
v_1 = lambda point: f_1(sym_v_1, (sym_d_1), d_1)(*point)
sym_v_1 = sp.expand(v_1(sym_xy_1))

# determine form of derivative
df_1 = lambda sym_function, sym_variable, point: sp.lambdify(sym_xy_1, sp.diff(sym_function, sym_variable))(*point)
epsilon_1 = lambda point: np.array([\
        [df_1(sym_u_1, sym_x_1, point), df_1(sym_u_1, sym_y_1, point) + df_1(sym_v_1, sym_x_1, point)], \
        [df_1(sym_u_1, sym_y_1, point) + df_1(sym_v_1, sym_x_1, point), df_1(sym_v_1, sym_y_1, point)], \
    ])

epsA_1 = epsilon_1(A0_1)
epsB_1 = epsilon_1(B0_1)
epsC_1 = epsilon_1(C0_1)
epsD_1 = epsilon_1(D0_1)

eps_1a = epsA_1 # m/m
eps_nO1a = np.linalg.norm(np.diag(eps_1a))
eps_tO1a = np.linalg.norm(eps_1a) - eps_nO1a

s_1 = lambda p, dc: np.matmul(p, dc)
t_1 = lambda p, s: np.sqrt(p**2 - s**2)
# i_1b, j_1b = -4, -3
# ijk_1b = np.array([i_1b, j_1b])
# s_1b = s_1(eps_1a, ijk_1b)
# dem_1b = np.sqrt(np.sum(np.power(ijk_1b, 2)))
dc_1b = l_1b, m_1b = np.cos(theta_1), np.sin(theta_1) # ijk_1b/dem_1b
# s_1b = s_1(eps_1a, dc_1b)
s_1b = np.array([\
        [(eps_1a[0][0] + eps_1a[1][1])/2 + (eps_1a[0][0] + eps_1a[1][1])/2*np.cos(2*theta_1) + eps_1a[0][1]/2*np.sin(2*theta_1), -(eps_1a[0][0] - eps_1a[1][1])*np.sin(2*theta_1) + eps_1a[0][1]*np.cos(2*theta_1)], \
        [-(eps_1a[0][0] - eps_1a[1][1])*np.sin(2*(theta_1 + np.pi/2)) + eps_1a[0][1]*np.cos(2*(theta_1 + np.pi/2)), (eps_1a[0][0] + eps_1a[1][1])/2 + (eps_1a[0][0] + eps_1a[1][1])/2*np.cos(2*(theta_1 + np.pi/2)) + eps_1a[0][1]/2*np.sin(2*(theta_1 + np.pi/2))], \
    ])
t_1b = t_1(np.linalg.norm(s_1b), s_1b)



A0_2, B0_2, C0_2, D0_2 = (0, 1), (1, 1), (1, 0), (0, 0) # m
A1_2, B1_2, C1_2, D1_2 = (0, 0.0125), (-0.0125, 0.0125), (0.025, 0.0125), (0, 0) # m
sym_c_2 = [sp.Symbol("c1"), sp.Symbol("c2"), sp.Symbol("c3"), sp.Symbol("c4")]
sym_d_2 = [sp.Symbol("d1"), sp.Symbol("d2"), sp.Symbol("d3"), sp.Symbol("d4")]
u_2 = lambda c, x, y: c[0] + c[1]*x + c[2]*y + c[3]*x*y
v_2 = lambda d, x, y: d[0] + d[1]*x + d[2]*y + d[3]*x*y
sym_xy_2 = sym_x_2, sym_y_2 = sp.Symbol("x"), sp.Symbol("y")
sym_u_2 = sp.expand(u_2(sym_c_2, *sym_xy_2))
sym_v_2 = sp.expand(v_2(sym_d_2, *sym_xy_2))

Auv_2 = np.array([\
        # at A
    [1, A0_2[0], A0_2[1], A0_2[0]*A0_2[1]], \
        # at B
    [1, B0_2[0], B0_2[1], B0_2[0]*B0_2[1]], \
        # at C
    [1, C0_2[0], C0_2[1], C0_2[0]*C0_2[1]], \
        # at D
    [1, D0_2[0], D0_2[1], D0_2[0]*D0_2[1]], \
    ])

bu_2 = np.array([A1_2[0], B1_2[0], C1_2[0], D1_2[0]]).reshape((len(Auv_2), 1))
x0u_2 = np.ones_like(bu_2)
c_2 = np.linalg.lstsq(Auv_2, bu_2)[0].reshape(1, len(x0u_2))[0]

bv_2 = np.array([A1_2[1], B1_2[1], C1_2[1], D1_2[1]]).reshape((len(Auv_2), 1))
x0v_2 = np.ones_like(bv_2)
d_2 = np.linalg.lstsq(Auv_2, bv_2)[0].reshape(1, len(x0v_2))[0]

# build string function from constants
f_2 = lambda sym_function, sym_variables, constants: sp.lambdify(sym_xy_2, sp.lambdify(sym_variables, sym_function)(*constants))
u_2 = lambda point: f_2(sym_u_2, (sym_c_2), c_2)(*point)
sym_u_2 = sp.expand(u_2(sym_xy_2))
v_2 = lambda point: f_2(sym_v_2, (sym_d_2), d_2)(*point)
sym_v_2 = sp.expand(v_2(sym_xy_2))

# determine form of derivative
df_2 = lambda sym_function, sym_variable, point: sp.lambdify(sym_xy_2, sp.diff(sym_function, sym_variable))(*point)
epsilon_2 = lambda point: np.array([\
        [df_2(sym_u_2, sym_x_2, point), df_2(sym_u_2, sym_y_2, point) + df_2(sym_v_2, sym_x_2, point)], \
        [df_2(sym_u_2, sym_y_2, point) + df_2(sym_v_2, sym_x_2, point), df_2(sym_v_2, sym_y_2, point)], \
    ])

epsA_2a = epsilon_2(A0_2)
epsB_2a = epsilon_2(B0_2)
epsC_2a = epsilon_2(C0_2)
epsD_2a = epsilon_2(D0_2)

eps_2a = epsA_2a # m/m
eps_nO2a = np.linalg.norm(np.diag(eps_2a))
eps_tO2a = np.linalg.norm(eps_2a) - eps_nO2a

green_strain_2 = lambda point: np.array([\
        [df_2(sym_u_2, sym_x_2, point) + 0.5*(df_2(sym_u_2, sym_x_2, point)**2 + df_2(sym_v_2, sym_x_2, point)**2), df_2(sym_v_2, sym_x_2, point) + df_2(sym_u_2, sym_y_2, point) + df_2(sym_u_2, sym_x_2, point)*df_2(sym_u_2, sym_y_2, point) + df_2(sym_v_2, sym_x_2, point)*df_2(sym_v_2, sym_y_2, point)], \
        [df_2(sym_v_2, sym_x_2, point) + df_2(sym_u_2, sym_y_2, point) + df_2(sym_u_2, sym_x_2, point)*df_2(sym_u_2, sym_y_2, point) + df_2(sym_v_2, sym_x_2, point)*df_2(sym_v_2, sym_y_2, point), df_2(sym_v_2, sym_y_2, point) + 0.5*(df_2(sym_u_2, sym_y_2, point)**2 + df_2(sym_v_2, sym_y_2, point)**2)], \
    ])
epsA_2b = green_strain_2(A0_2)
epsB_2b = green_strain_2(B0_2)
epsC_2b = green_strain_2(C0_2)
epsD_2b = green_strain_2(D0_2)

eps_2b = epsA_2b # m/m

s_2 = lambda p, dc: p*dc#np.matmul(p, dc)
t_2 = lambda p, s: np.sqrt(p**2 - s**2)
i_2c, j_2c = 1, 1
ijk_2c = np.array([i_2c, j_2c])
eps_2c = s_2(eps_2b, ijk_2c)
dem_2c = np.sqrt(np.sum(np.power(ijk_2c, 2)))
dc_2c = l_2c, m_2c = ijk_2c/dem_2c
s_2c = s_2(eps_2b, dc_2c)
t_2c = t_2(np.linalg.norm(s_2c), s_2c)

norm_a = nm.norms(eps_2a).l_two()
norm_b = nm.norms(eps_2b).l_two()
norm_avg = (norm_a + norm_b)/2
perc_err_2d = np.abs((norm_a - norm_b)/norm_avg)

norm_c = nm.norms(s_2c).l_two()
perc_err_2e = np.abs((norm_a - norm_c)/norm_avg)



c_3 = 1e-4
point_3 = x_3, y_3, z_3 = 0, 2, 1 # m
u_3 = lambda x, y, z: (x**2 + 10)*c_3
v_3 = lambda x, y, z: 2*(y*z)*c_3
w_3 = lambda x, y, z: (z**2 - x*y)*c_3
variables_3 = sym_x_3, sym_y_3, sym_z_3 = sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")
sym_u_3 = sp.expand(u_3(*variables_3))
sym_v_3 = sp.expand(v_3(*variables_3))
sym_w_3 = sp.expand(w_3(*variables_3))
# determine form of derivative
df_3 = lambda sym_function, variable, point: sp.lambdify(variables_3, sp.diff(sym_function, variable))(*point)
epsilon_3 = np.array([\
        [df_3(sym_u_3, sym_x_3, point_3), df_3(sym_u_3, sym_y_3, point_3) + df_3(sym_v_3, sym_x_3, point_3), df_3(sym_w_3, sym_x_3, point_3) + df_3(sym_u_3, sym_z_3, point_3)], \
        [0, df_3(sym_v_3, sym_y_3, point_3), df_3(sym_v_3, sym_z_3, point_3) + df_3(sym_w_3, sym_y_3, point_3)], \
        [0, 0, df_3(sym_w_3, sym_z_3, point_3)], \
    ])



A0_5, B0_5, C0_5, D0_5 = (0, 1), (1, 1), (1, 0), (0, 0) # m
A1_5, B1_5, C1_5, D1_5 = (0, 0.0125), (-0.0125, 0.0125), (0.025, 0.0125), (0, 0) # m
sym_c_5 = [sp.Symbol("c1"), sp.Symbol("c2"), sp.Symbol("c3"), sp.Symbol("c4")]
sym_d_5 = [sp.Symbol("d1"), sp.Symbol("d2"), sp.Symbol("d3"), sp.Symbol("d4")]
u_5 = lambda c, x, y: c[0] + c[1]*x + c[2]*y + c[3]*x*y
v_5 = lambda d, x, y: d[0] + d[1]*x + d[2]*y + d[3]*x*y
sym_xy_5 = sym_x_5, sym_y_5 = sp.Symbol("x"), sp.Symbol("y")
sym_u_5 = sp.expand(u_5(sym_c_5, *sym_xy_5))
sym_v_5 = sp.expand(v_5(sym_d_5, *sym_xy_5))

Auv_5 = np.array([\
        # at A
    [1, A0_5[0], A0_5[1], A0_5[0]*A0_5[1]], \
        # at B
    [1, B0_5[0], B0_5[1], B0_5[0]*B0_5[1]], \
        # at C
    [1, C0_5[0], C0_5[1], C0_5[0]*C0_5[1]], \
        # at D
    [1, D0_5[0], D0_5[1], D0_5[0]*D0_5[1]], \
    ])

bu_5 = np.array([A1_5[0], B1_5[0], C1_5[0], D1_5[0]]).reshape((len(Auv_5), 1))
x0u_5 = np.ones_like(bu_5)
c_5 = np.linalg.lstsq(Auv_5, bu_5)[0].reshape(1, len(x0u_5))[0]

bv_5 = np.array([A1_5[1], B1_5[1], C1_5[1], D1_5[1]]).reshape((len(Auv_5), 1))
x0v_5 = np.ones_like(bv_5)
d_5 = np.linalg.lstsq(Auv_5, bv_5)[0].reshape(1, len(x0v_5))[0]

# build string function from constants
f_5 = lambda sym_function, sym_variables, constants: sp.lambdify(sym_xy_5, sp.lambdify(sym_variables, sym_function)(*constants))
u_5 = lambda point: f_5(sym_u_5, (sym_c_5), c_5)(*point)
sym_u_5 = sp.expand(u_5(sym_xy_5))
v_5 = lambda point: f_5(sym_v_5, (sym_d_5), d_5)(*point)
sym_v_5 = sp.expand(v_5(sym_xy_5))

# determine form of derivative
df_5 = lambda sym_function, sym_variable, point: sp.lambdify(sym_xy_5, sp.diff(sym_function, sym_variable))(*point)
epsilon_5 = lambda point: np.array([\
        [df_5(sym_u_5, sym_x_5, point), df_5(sym_u_5, sym_y_5, point) + df_5(sym_v_5, sym_x_5, point), 0], \
        [df_5(sym_u_5, sym_y_5, point) + df_5(sym_v_5, sym_x_5, point), df_5(sym_v_5, sym_y_5, point), 0], \
        [0, 0, 0] \
    ])

epsA_5 = epsilon_5(A0_5)
epsB_5 = epsilon_5(B0_5)
epsC_5 = epsilon_5(C0_5)
epsD_5 = epsilon_5(D0_5)

eps_5 = epsA_5 # m/m
dir_5 = 0.5*np.arctan(eps_5[0][1]/(eps_5[0][0] - eps_5[1][1])) # rad
j1_5 = np.trace(eps_5)
j2_5 = eps_5[0][0]*eps_5[1][1] + eps_5[0][0]*eps_5[2][2] + eps_5[1][1]*eps_5[2][2] - eps_5[0][1]**2 - eps_5[2][1]**2 - eps_5[0][2]**2
j3_5 = sp.det(sp.Matrix(eps_5))
eps_sym = sp.Symbol("eps")
f_5 = eps_sym**3 - j1_5*eps_sym**2 - j2_5*eps_sym - j3_5
principals_5 = []
for p in sp.solve(f_5, eps_sym): principals_5.append(complex(p))
principals_5 = np.sort(np.abs(principals_5))[::-1]
mag_5 = 0
for p in principals_5:
    if np.abs(p) >= mag_5:
        mag_5 = np.abs(p)
t_max_5 = np.average(principals_5[:1])



a0_6, b0_6 = 20, 12 # mm, mm
eps_6 = np.array([\
        [300, 200, 0], \
        [200, 500, 0], \
        [0, 0, 0], \
    ])/1e6 # strain
AC0_6 = QB0_6 = np.sqrt(a0_6**2 + b0_6**2)
delta_a_6, delta_b_6 = eps_6[0][0]*a0_6, eps_6[1][1]*b0_6 # mm, mm
a1_6, b1_6 = a0_6 + delta_a_6, b0_6 + delta_b_6
AC1_6 = QB1_6 = np.sqrt(a1_6**2 + b1_6**2)
deltaAC_6 = deltaQB_6 = AC1_6 - AC0_6



eps_7 = np.array([\
    [400, 100, 0], \
    [100, 0, -200], \
    [0, -200, 600], \
    ])/1e6
n_7 = lambda p, dc: np.matmul(p, dc)
t_7 = lambda p, s: np.sqrt(p**2 - s**2)

j1_7 = np.trace(eps_7) # MPa
j2_7 = eps_7[0][0]*eps_7[1][1] + eps_7[0][0]*eps_7[2][2] + eps_7[1][1]*eps_7[2][2] - eps_7[0][1]**2 - eps_7[2][1]**2 - eps_7[0][2]**2 # MPa**2
j3_7 = sp.det(sp.Matrix(eps_7)) # MPa**3
eps_sym = sp.Symbol("eps")
f_7 = eps_sym**3 - j1_7*eps_sym**2 - j2_7*eps_sym - j3_7
principals_7 = []
for p in sp.solve(f_7, eps_sym): principals_7.append(complex(p))
mag_7 = 0
for p in principals_7:
    if np.abs(p) >= mag_7:
        mag_7 = np.abs(p)
        dir_7 = np.arctan(p.imag/p.real)
t_max_7 = np.average(np.sort(np.abs(principals_7))[:1])

theta1_b = np.radians(30) # rad
i_7b, j_7b, k_7b = 1, 1, 1
i_7b, j_7b, k_7b = i_7b*np.cos(theta1_b), j_7b*np.sin(theta1_b), k_7b
ijk_7b = np.array([i_7b, j_7b, k_7b])
eps_7b = n_7(eps_7, ijk_7b)
dem_7b = np.sqrt(np.sum(np.power(ijk_7b, 2)))
dc_7b = l_7b, m_7b, n_7b = ijk_7b/dem_7b
s_7b = n_7(eps_7b, dc_7b)
t_7b= t_7(np.linalg.norm(eps_7b), s_7b)



eps_8 = np.array([\
    [-300, -583, -300], \
    [-583, 200, -67], \
    [-300, -67, -200], \
    ])/1e6
n_8 = lambda p, dc: np.matmul(p, dc)
t_8 = lambda p, s: np.sqrt(p**2 - s**2)

j1_8 = np.trace(eps_8) # MPa
j2_8 = eps_8[0][0]*eps_8[1][1] + eps_8[0][0]*eps_8[2][2] + eps_8[1][1]*eps_8[2][2] - eps_8[0][1]**2 - eps_8[2][1]**2 - eps_8[0][2]**2 # MPa**2
j3_8 = sp.det(sp.Matrix(eps_8)) # MPa**3
eps_sym = sp.Symbol("eps")
f_8 = eps_sym**3 - j1_8*eps_sym**2 - j2_8*eps_sym - j3_8
principals_8 = []
for p in sp.solve(f_8, eps_sym): principals_8.append(complex(p))
mag_8 = 0
for p in principals_8:
    if np.abs(p) >= mag_8:
        mag_8 = np.abs(p)
        dir_8 = np.arctan(p.imag/p.real)
t_max_8 = np.average(np.sort(np.abs(principals_8))[:1])



point_10 = x_10, y_10, z_10 = 3/4, 1/4, 1/2 # mm
E_10, nu_10 = 200e3, 0.25 # MPa, ~
G_10 = E_10/(2*(1 + nu_10)) # MPa
sx_10 = lambda x, y, z: -(x**3) + y**2
sy_10 = lambda x, y, z: 2*(x**2) + (y**2)/2
sz_10 = lambda x, y, z: 4*(y**2) - z**3
txy_10 = lambda x, y, z: 5*z + 2*(y**2)
txz_10 = lambda x, y, z: x*(z**3) + (x**2)*y
tyz_10 = lambda x, y, z: 0
f_10 = lambda point: np.array([\
        [(sx_10(*point) - nu_10*(sy_10(*point) + sz_10(*point)))/E_10, txy_10(*point)/G_10, txz_10(*point)/G_10], \
        [0, (sy_10(*point) - nu_10*(sx_10(*point) + sz_10(*point)))/E_10, tyz_10(*point)/G_10], \
        [0, 0, (sz_10(*point) - nu_10*(sx_10(*point) + sy_10(*point)))/E_10], \
    ])
epsilon_10 = f_10(point_10)



def strain_tensor_11(sigma, E, nu):
    G = E/(2*(1 + nu))
    e = np.trace(sigma)*(1 - 2*nu)/E
    strain = np.diagflat(sigma.diagonal()/E)
    strain -= np.array([\
        [sigma[1][1] + sigma[2][2]], \
        [sigma[0][0] + sigma[2][2]], \
        [sigma[0][0] + sigma[1][1]], \
        ])*nu/E
    strain += np.diagflat(np.diag(sigma, k=1)/G, k=1)
    strain += np.diagflat(np.diag(sigma, k=2)/G, k=2)
    return G, e, strain

a0_11, b0_11, t0_11 = 300, 400, 10 # mm, mm, mm
E_11, nu_11 = 70e3, 1/3 # MPa, ~
sigma_11 = np.array([\
        [30, 0, 0], \
        [0, 90, 0], \
        [0, 0, 0], \
    ])
V0_11 = a0_11*b0_11*t0_11 # mm3
G_11, e_11, epsilon_11 = strain_tensor_11(sigma_11, E_11, nu_11)
delta_b_11, deltaV1_11 = epsilon_11[1][1]*b0_11, e_11*V0_11 # mm, mm3



def strain_tensor_12(sigma, E, nu):
    G = E/(2*(1 + nu))
    e = np.trace(sigma)*(1 - 2*nu)/E
    strain = np.diagflat(sigma.diagonal()/E)
    strain -= np.array([\
        [sigma[1][1] + sigma[2][2]], \
        [sigma[0][0] + sigma[2][2]], \
        [sigma[0][0] + sigma[1][1]], \
        ])*nu/E
    strain += np.diagflat(np.diag(sigma, k=1)/G, k=1)
    strain += np.diagflat(np.diag(sigma, k=2)/G, k=2)
    return G, e, strain

a0_12, b0_12, c0_12 = D0_12 = np.array([250, 200, 150]) # mm, mm, mm
E_12, nu_12 = 200e3, 0.3 # MPa, ~
sigma_12 = np.array([\
        [-60, 0, 0], \
        [0, -50, 0], \
        [0, 0, -40], \
    ])
V0_12 = a0_12*b0_12*c0_12 # mm3
G_12, e_12, epsilon_12 = strain_tensor_12(sigma_12, E_12, nu_12)
delta_a_12, delta_b_12, delta_c_12 = np.matmul(epsilon_12, D0_12) # mm
deltaV1_12 = e_12*V0_12 # mm3


def render_body(context,**pageargs):
    __M_caller = context.caller_stack._push_frame()
    try:
        __M_locals = __M_dict_builtin(pageargs=pageargs)
        round = context.get('round', UNDEFINED)
        __M_writer = context.writer()
        __M_writer('// document metadata\r\n= ENGR 727-001 Advanced Mechanics of Materials: Homework 4\r\nJoby M. Anthony III <jmanthony1@liberty.edu>\r\n:affiliation: PhD Student\r\n:document_version: 1.0\r\n:revdate: February 16, 2022\r\n// :description: \r\n// :keywords: \r\n:imagesdir: ./ENGR727_Homework4_JobyAnthonyIII\r\n:bibtex-file: ENGR727_Homework4_JobyAnthonyIII.bib\r\n:toc: auto\r\n:xrefstyle: short\r\n:sectnums: |,all|\r\n:chapter-refsig: Chap.\r\n:section-refsig: Sec.\r\n:stem: latexmath\r\n:eqnums: AMS\r\n:stylesdir: C:/Users/jmanthony1/Documents/GitHub/WeCANDoIt/Asciidoc/Testing/ENGR527-727 HW4\r\n:stylesheet: asme.css\r\n:noheader:\r\n:nofooter:\r\n:docinfo: private\r\n:docinfodir: C:/Users/jmanthony1/Documents/GitHub/WeCANDoIt/Asciidoc/Testing/ENGR527-727 HW4\r\n:front-matter: any\r\n:!last-update-label:\r\n\r\n// example variable\r\n// :fn-1: footnote:[]\r\n\r\n// Python modules\r\n')
        __M_writer('\r\n// end document metadata\r\n\r\n\r\n\r\n\r\n\r\n// begin document\r\n// [abstract]\r\n// .Abstract\r\n\r\n// // *Keywords:* _{keywords}_\r\n\r\nProblems:\r\n\r\n* [x] Problem 1\r\n* [x] Problem 2\r\n* [x] Problem 3\r\n* [x] Problem 4\r\n* [x] Problem 5\r\n* [x] Problem 6\r\n* [x] Problem 7\r\n* [x] Problem 8\r\n* [ ] Problem 9\r\n* [x] Problem 10\r\n* [x] Problem 11\r\n* [x] Problem 12\r\n\r\n\r\n\r\n[#sec-1, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 1\r\n:subs: 0\r\n:figs: 0\r\n\r\n> A square, glass block in the side of a skyscraper is loaded so that the block is in a state of plane strain (stem:[\\epsilon_{zx} = \\epsilon_{zy} = \\epsilon_{zz} = 0]).\r\n> (a) Determine the displacements for the block for the deformations shown and the strain components for the stem:[xy]-coordinate axes.\r\n> (b) Determine the strain components for the stem:[XY]-axes.\r\n> [#fig-1-problem_statement]\r\n> .Adapted from assignment instructions.\r\n> image::./1-problem_statement_220221_192020_EST.png[caption=<span class="figgynumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n> -- Problem Statement\r\n\r\n.Solution\r\n// solution codes\r\n')
        __M_writer('\r\n\r\nThe displacement for the block are as shown in xref:fig-1-problem_statement[].\r\nThe strain components along the stem:[xy]-axes can be found from building the strain state in matrix notation:\r\n[stem#eq-1-strain_state_form, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n[\\epsilon] = \\begin{bmatrix}\\frac{\\partial u}{\\partial x} & \\frac{\\partial u}{\\partial y} + \\frac{\\partial v}{\\partial x} \\\\')
        __M_writer('\\frac{\\partial u}{\\partial y} + \\frac{\\partial v}{\\partial x} & \\frac{\\partial v}{\\partial y}\\end{bmatrix}\r\n\\end{equation}\r\n++++\r\nEquations stem:[u(x, y)] and stem:[v(x, y)] can be found from solving a linear system of equations for which there are four equations and four unknowns:\r\n\r\n[stem#eq-1-equation_forms, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{split}\r\nu(x, y) &:= \\begin{bmatrix}1 & x_{1} & y_{1} & xy_{1} \\\\')
        __M_writer('1 & x_{2} & y_{2} & xy_{2} \\\\')
        __M_writer('1 & x_{3} & y_{3} & xy_{3} \\\\')
        __M_writer('1 & x_{4} & y_{4} & xy_{4}\\end{bmatrix}\\begin{bmatrix}c_{1} \\\\')
        __M_writer('c_{2} \\\\')
        __M_writer('c_{3} \\\\')
        __M_writer('c_{4}\\end{bmatrix} &= \\begin{bmatrix}\\delta_{x1} \\\\')
        __M_writer('\\delta_{x2} \\\\')
        __M_writer('\\delta_{x3} \\\\')
        __M_writer('\\delta_{x4}\\end{bmatrix} \\\\')
        __M_writer('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\r\nd(x, y) &:= \\begin{bmatrix}1 & x_{1} & y_{1} & xy_{1} \\\\')
        __M_writer('1 & x_{2} & y_{2} & xy_{2} \\\\')
        __M_writer('1 & x_{3} & y_{3} & xy_{3} \\\\')
        __M_writer('1 & x_{4} & y_{4} & xy_{4}\\end{bmatrix}\\begin{bmatrix}d_{1} \\\\')
        __M_writer('d_{2} \\\\')
        __M_writer('d_{3} \\\\')
        __M_writer('d_{4}\\end{bmatrix} &= \\begin{bmatrix}\\delta_{y1} \\\\')
        __M_writer('\\delta_{y2} \\\\')
        __M_writer('\\delta_{y3} \\\\')
        __M_writer('\\delta_{y4}\\end{bmatrix}\r\n\\end{split}\r\n\\end{equation}\r\n++++\r\n\r\nSolving xref:eq-1-equation_forms[] with input data from the appropriate points lends to xref:eq-1-equations[].\r\n[stem#eq-1-equations, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{split}\r\nu(x, y) &= ')
        __M_writer(str(sym_u_1))
        __M_writer(' \\\\')
        __M_writer('v(x, y) &= ')
        __M_writer(str(sym_v_1))
        __M_writer('\r\n\\end{split}\r\n\\end{equation}\r\n++++\r\nThis means that xref:eq-1-strain_state_form[] can now be solved to reveal the strain state of the glass block: stem:[[\\epsilon_{xy}\\] = \\begin{bmatrix}')
        __M_writer(str(engr(eps_1a[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(eps_1a[0][1])))
        __M_writer(' \\\\')
        __M_writer(str(engr(eps_1a[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(eps_1a[1][1])))
        __M_writer('\\end{bmatrix}].\r\n\r\nTo find the strain components projected onto the stem:[XY]-axes, which is at some angle offset from the stem:[xy]-axes, xref:eq-1-strain_offset[] must be implemented.\r\n[stem#eq-1-strain_offset, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{split}\r\n\\epsilon_{x\'} &= \\epsilon_{x}\\cos^{2}(\\theta) + \\epsilon_{y}\\sin^{2}(\\theta) + \\gamma_{xy}\\sin(\\theta)\\cos(\\theta) \\\\')
        __M_writer("\\gamma_{x'y'} &= -(\\epsilon_{x} - \\epsilon_{y})\\sin(2\\theta) + \\gamma_{xy}\\cos(2\\theta)\r\n\\end{split}\r\n\\end{equation}\r\n++++\r\n, wherein stem:[\\epsilon_{y'}] can be found by replacing stem:[\\theta] with stem:[\\theta + \\frac{\\pi}{2}] in the equation for stem:[\\epsilon_{x'}].\r\nThis yields stem:[[\\epsilon_{XY}\\] = \\begin{bmatrix}")
        __M_writer(str(engr(s_1b[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(s_1b[0][1])))
        __M_writer(' \\\\')
        __M_writer(str(engr(s_1b[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(s_1b[1][1])))
        __M_writer('\\end{bmatrix}]\r\n\r\n\r\n\r\n[#sec-2, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 2\r\n:subs: 0\r\n:figs: 0\r\n\r\n> A square plate, stem:[1~m] on a long side, is loaded in a state of plane strain and is deformed as shown.\r\n> (a) Write expressions for the stem:[u] and stem:[v] displacements for any point on the plate.\r\n> (b) Determine the components of *Green Strain* in the plate.\r\n> (c) Determine the total *Green Strain* at point stem:[B] for a line element in the direction of line stem:[OB].\r\n> (d) For point stem:[B], compare the components of strain from part (b) to the components of strain for *Small-Displacement Theory*.\r\n> (e) Compare the strain determined in part (c) to the corresponding strain using *Small-Displacement Theory*.\r\n> [#fig-2-problem_statement]\r\n> .Adapted from assignment instructions.\r\n> image::./2-problem_statement_220221_192420_EST.png[caption=<span class="figgynumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n> -- Problem Statement\r\n\r\n.Solution\r\n// solution codes\r\n')
        __M_writer('\r\n\r\nAs demonstrated in xref:sec-1[], expressions for displacement for any point on the plate may be found by solving xref:eq-1-equation_forms[].\r\nThis yields the following expressions:\r\n[stem]\r\n++++\r\n\\begin{split}\r\nu(x, y) &= ')
        __M_writer(str(sym_u_2))
        __M_writer(' \\\\')
        __M_writer('v(x, y) &= ')
        __M_writer(str(sym_v_2))
        __M_writer('\r\n\\end{split}\r\n++++\r\nThe strain state of the plate may be found by solving xref:eq-2-green_strain_form[] with the developed displacement equations.\r\n[stem#eq-2-green_strain_form, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{split}\r\n\\epsilon_{x} &= \\frac{\\partial u}{\\partial x} + \\frac{1}{2}\\Bigl[\\Bigl(\\frac{\\partial u}{\\partial x}\\Bigr)^{2} + \\Bigl(\\frac{\\partial v}{\\partial x}\\Bigr)^{2}\\Bigr] \\\\')
        __M_writer('\\epsilon_{y} &= \\frac{\\partial v}{\\partial y} + \\frac{1}{2}\\Bigl[\\Bigl(\\frac{\\partial u}{\\partial y}\\Bigr)^{2} + \\Bigl(\\frac{\\partial v}{\\partial y}\\Bigr)^{2}\\Bigr] \\\\')
        __M_writer('\\gamma_{xy} &= \\frac{\\partial v}{\\partial x} + \\frac{\\partial u}{\\partial y} + \\frac{\\partial u}{\\partial x}\\frac{\\partial u}{\\partial y} + \\frac{\\partial v}{\\partial x}\\frac{\\partial v}{\\partial y}\r\n\\end{split}\r\n\\end{equation}\r\n++++\r\nThe strain state, then, becomes: stem:[\\epsilon_{Green} = \\begin{bmatrix}')
        __M_writer(str(engr(eps_2b[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(eps_2b[0][1])))
        __M_writer(' \\\\\\ ')
        __M_writer(str(engr(eps_2b[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(eps_2b[1][1])))
        __M_writer('\\end{bmatrix}].\r\nThe percent error between stem:[\\epsilon_{Green}] and stem:[\\epsilon_{Small} = \\begin{bmatrix}')
        __M_writer(str(engr(eps_2a[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(eps_2a[0][1])))
        __M_writer(' \\\\\\ ')
        __M_writer(str(engr(eps_2a[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(eps_2a[1][1])))
        __M_writer('\\end{bmatrix}] is stem:[')
        __M_writer(str(round(perc_err_2d, 4)))
        __M_writer('~\\%].\r\nProjecting this strain state along the line stem:[OB] can be found by the _direction of cosines_ for this stem:[<1, 1>] vector (stem:[l = ')
        __M_writer(str(round(l_2c, 4)))
        __M_writer('] and stem:[m = ')
        __M_writer(str(round(m_2c, 4)))
        __M_writer(']): stem:[(\\epsilon_{Green})_{OB} = \\begin{bmatrix}')
        __M_writer(str(engr(s_2c[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(s_2c[0][1])))
        __M_writer(' \\\\\\ ')
        __M_writer(str(engr(s_2c[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(s_2c[1][1])))
        __M_writer('\\end{bmatrix}].\r\nThe percent error between stem:[\\epsilon_{Small}] and stem:[(\\epsilon_{Green})_{OB}] is stem:[')
        __M_writer(str(round(perc_err_2e, 4)))
        __M_writer('~\\%].\r\n\r\n\r\n\r\n[#sec-3, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 3\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.3 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.3\r\nA displacement field in a body is given by\r\n[stem#eq-3-problem_statement, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{split}\r\nu = c(x^{2} + 10), &\\quad v = 2cyz, &\\quad w = c(-xy + z^{2})\r\n\\end{split}\r\n\\end{equation}\r\n++++\r\nwhere stem:[c = 10^{-4}].\r\nDetermine the state of strain on an element positioned at stem:[(0, 2, 1)].\r\n\r\n.Solution\r\n// solution codes\r\n')
        __M_writer('\r\n\r\nThe strain state can be found from determining the strain state at the point in the displacement field:\r\n[stem#eq-3-strain_state, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n[\\epsilon] = \\begin{bmatrix}\\epsilon_{x} = \\frac{\\partial u}{\\partial x} & \\gamma_{xy} = \\frac{\\partial u}{\\partial y} + \\frac{\\partial v}{\\partial x} & \\gamma_{xz} = \\frac{\\partial w}{\\partial x} + \\frac{\\partial u}{\\partial z} \\\\')
        __M_writer('0 & \\epsilon_{y} = \\frac{\\partial v}{\\partial y} & \\gamma_{yz} = \\frac{\\partial v}{\\partial z} + \\frac{\\partial w}{\\partial y} \\\\')
        __M_writer('0 & 0 & \\epsilon_{z} = \\frac{\\partial w}{\\partial z}\\end{bmatrix}\r\n\\end{equation}\r\n++++\r\nWhen xref:eq-3-problem_statement[] is plugged into xref:eq-3-strain_state[], this takes the form\r\n[stem]\r\n++++\r\n[\\epsilon] = \\begin{bmatrix}2x & 0 & -y \\\\')
        __M_writer('0 & 2z & 2y - x \\\\')
        __M_writer('0 & 0 & 2z\\end{bmatrix}\\times 10^{-4}\r\n++++\r\nwhich further yields xref:eq-5-strain_p[] for stem:[\\mathbf{\\epsilon}(x = 0, y = 2, z = 1)].\r\n[stem#eq-3-strain_p, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n[\\epsilon] = \\begin{bmatrix}2 & 0 & -1 \\\\')
        __M_writer('0 & 4 & 1 \\\\')
        __M_writer('0 & 0 & 4\\end{bmatrix}\\times 10^{-4}\r\n\\end{equation}\r\n++++\r\n\r\n.Answer\r\nThe strain tensor at point stem:[(0, 2, 1)], stem:[\\epsilon = \\begin{bmatrix}')
        __M_writer(str(engr(epsilon_3[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_3[0][1])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_3[0][2])))
        __M_writer(' \\\\')
        __M_writer(str(engr(epsilon_3[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_3[1][1])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_3[1][2])))
        __M_writer(' \\\\')
        __M_writer(str(engr(epsilon_3[2][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_3[2][1])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_3[2][2])))
        __M_writer('\\end{bmatrix}].\r\n\r\n\r\n\r\n[#sec-4, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 4\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.4 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.4\r\nThe displacement field and strain distribution in a member have the form\r\n[stem#eq-4-problem_statement, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{split}\r\nu &= a_{0}x^{2}y^{2} + a_{1}xy^{2} + a_{2}x^{2}y \\\\')
        __M_writer('v &= b_{0}x^{2}y + b_{1}xy \\\\')
        __M_writer('\\gamma_{xy} &= c_{0}x^{2}y + c_{1}xy + c_{2}x^{2} + c_{3}y^{2}\r\n\\end{split}\r\n\\end{equation}\r\n++++\r\nWhich relationships connecting the constraints (stem:[a]\'s, stem:[b]\'s, and stem:[c]\'s) make the foregoing expressions (xref:eq-4-problem_statement[]) possible?\r\n\r\n.Answer\r\nThese equations assume _plane-strain_; therefore, the constants can be found from the contribution of the respective derivatives of each displacement field equation.\r\n\r\n\r\n\r\n[#sec-5, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 5\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.9 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.9\r\nA stem:[100~mm] by stem:[150~mm] rectangular plate stem:[QABC] is deformed into the shape shown by the dashed lines in xref:fig-5-problem_statement[].\r\nAll dimensions shown in the figure are in millimeters.\r\nDetermine at point stem:[Q] (a) the strain components stem:[\\epsilon_{x}], stem:[\\epsilon_{y}], and stem:[\\gamma_{xy}] and (b) the principal strains and the direction of the principal axes.\r\n[#fig-5-problem_statement]\r\n.Adapted from cite:[uguralAdvancedMechanicsMaterials2019].\r\nimage::./5-problem_statement_220221_194927_EST.png[caption=<span class="figgynumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n\r\n.Solution\r\n// solution codes\r\n')
        __M_writer('\r\n\r\nFollowing a procedure similar to that displayed in xref:sec-1[] yields the strain state: stem:[\\begin{bmatrix}')
        __M_writer(str(engr(eps_5[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(eps_5[0][1])))
        __M_writer(' \\\\\\ ')
        __M_writer(str(engr(eps_5[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(eps_5[1][1])))
        __M_writer('\\end{bmatrix}] which yields the principal strains: stem:[\\epsilon_{1}, \\epsilon_{2}, \\epsilon_{3} = ')
        __M_writer(str(engr(principals_5[0])))
        __M_writer(', ')
        __M_writer(str(engr(principals_5[1])))
        __M_writer(', ')
        __M_writer(str(engr(principals_5[2])))
        __M_writer('] in the stem:[\\theta_{p} = ')
        __M_writer(str(engr(dir_5)))
        __M_writer('~rad] direction.\r\n\r\n\r\n\r\n[#sec-6, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 6\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.12 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.12\r\nA thin, rectangular plate stem:[a = 20~mm \\times b = 12~mm] (xref:fig-6-problem_statement[]) is acted upon by a stress distribution resulting in the uniform strains stem:[\\epsilon_{x} = 300\\mu], stem:[\\epsilon_{y} = 500\\mu], and stem:[\\gamma_{xy} = 200\\mu].\r\nDetermine the changes in length of diagonals stem:[QB] and stem:[AC].\r\n[#fig-6-problem_statement]\r\n.Adapted from cite:[uguralAdvancedMechanicsMaterials2019].\r\nimage::./6-problem_statement_220221_195504_EST.png[caption=<span class="figgynumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n\r\n.Solution\r\n// solution codes\r\n')
        __M_writer("\r\n\r\nBecause the plate undergoes a displacement that yields a symmetric strain state, the change in length for diagonals stem:[AC] and stem:[QB] are equal and found by finding the difference of the hypotenuse: e.g. stem:[a' = (1 + \\epsilon_{x})a_{0} = ")
        __M_writer(str(engr(a1_6)))
        __M_writer('].\r\nTherefore, stem:[\\Delta l_{AC} = ')
        __M_writer(str(engr(deltaAC_6)))
        __M_writer('~m] and stem:[\\Delta l_{QB} = ')
        __M_writer(str(engr(deltaQB_6)))
        __M_writer('~m].\r\n\r\n\r\n\r\n[#sec-7, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 7\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.22 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.22\r\nSolve Problem 2.21 cite:[uguralAdvancedMechanicsMaterials2019] for a state of strain given by\r\n[stem#eq-7-problem_statement, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{bmatrix}400 & 100 & 0 \\\\')
        __M_writer('100 & 0 & -200 \\\\')
        __M_writer("0 & -200 & 600\\end{bmatrix}~\\mu\r\n\\end{equation}\r\n++++\r\nProblem 2.21 asks to determine (a) the strain invariants; (b) the normal strain in the stem:[x'] direction which is directed at an angle stem:[\\theta = 30~^{\\circ}] from the stem:[x]-axis; (c) the principal strains stem:[\\epsilon_{1}], stem:[\\epsilon_{2}], and stem:[\\epsilon_{3}]; and, (d) the maximum shear strain.\r\n\r\n.Solution\r\n// solution codes\r\n")
        __M_writer('\r\n// i_2a, j_2a, k_2a = 2, 1, 2\r\n// ijk_2a = np.array([i_2a, j_2a, k_2a])\r\n// eps_7a = s_2(eps_7, ijk_2a)\r\n// dem_2a = np.sqrt(np.sum(np.power(ijk_2a, 2)))\r\n// dc_2a = l_2a, m_2a, n_2a = ijk_2a/dem_2a\r\n// s_2a = s_2(eps_7a, dc_2a)\r\n// t_2a = t_2(np.linalg.norm(eps_7a), s_2a)\r\n\r\n\r\n.Answers\r\nThe strain invariants of given strain state (xref:eq-7-problem_statement[]):\r\n[stem]\r\n++++\r\n\\begin{split}\r\nJ_{1} &= p_{x} + p_{y} + p_{z} = ')
        __M_writer(str(engr(j1_7)))
        __M_writer(' \\\\')
        __M_writer('J_{2} &= p_{x}p_{y} + p_{x}p_{z} + p_{y}p_{z} \\\\')
        __M_writer(' &\\quad- p_{xy}^{2} - p_{yz}^{2} - p_{xz}^{2} \\\\')
        __M_writer('J_{2} &= ')
        __M_writer(str(engr(j2_7)))
        __M_writer(' \\\\')
        __M_writer('J_{3} &= \\|\\mathbf{p}\\| = ')
        __M_writer(str(round(j3_7, 4)))
        __M_writer("\r\n\\end{split}\r\n++++\r\n\r\nThe normal strain along stem:[x'], which is stem:[\\theta = 30~^{\\circ}] up from the stem:[x]-axis, is stem:[\\epsilon_{x'} = ")
        __M_writer(str(engr(s_7b)))
        __M_writer('].\r\n\r\nThe principal strains come from solving stem:[\\epsilon_{p}^{3} - J_{1}\\epsilon_{p}^{2} + J_{2}\\epsilon_{p} - J_{3} = 0]:\r\n\r\n* stem:[\\epsilon_{1} = ')
        __M_writer(str(engr(np.sort(np.abs(principals_7))[::-1][0])))
        __M_writer(']\r\n* stem:[\\epsilon_{2} = ')
        __M_writer(str(engr(np.sort(np.abs(principals_7))[::-1][1])))
        __M_writer(']\r\n* stem:[\\epsilon_{3} = ')
        __M_writer(str(engr(np.sort(np.abs(principals_7))[::-1][2])))
        __M_writer(']\r\n\r\nThe maximum principal strain (magnitude and direction), stem:[\\epsilon_{1} = ')
        __M_writer(str(engr(mag_7)))
        __M_writer('~\\angle~')
        __M_writer(str(engr(dir_7)))
        __M_writer("~rad].\r\nThe magnitude of the shear strain is the average of the principal strains (from *Mohr's Circle*): stem:[\\gamma_{max} = ")
        __M_writer(str(engr(t_max_7)))
        __M_writer('].\r\n\r\n\r\n\r\n[#sec-8, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 8\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.24 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.24\r\nAt a point in a loaded frame, the strain with respect to the coordinate set stem:[xyz] is\r\n[stem#eq-8-problem_statement, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{bmatrix}-300 & -583 & -300 \\\\')
        __M_writer('-583 & 200 & -67 \\\\')
        __M_writer('-300 & -67 & -200\\end{bmatrix}~\\mu\r\n\\end{equation}\r\n++++\r\nDetermine (a) the magnitudes and directions of the principal strains and (b) the maxmimum shear strains.\r\n\r\n.Solution\r\n// solution codes\r\n')
        __M_writer('\r\n\r\n.Answers\r\nThe principal strains come from solving stem:[\\epsilon_{p}^{3} - J_{1}\\epsilon_{p}^{2} + J_{2}\\epsilon_{p} - J_{3} = 0]:\r\nThe strain invariants of given strain state (xref:eq-8-problem_statement[]):\r\n[stem]\r\n++++\r\n\\begin{split}\r\nJ_{1} &= p_{x} + p_{y} + p_{z} = ')
        __M_writer(str(engr(j1_8)))
        __M_writer(' \\\\')
        __M_writer('J_{2} &= p_{x}p_{y} + p_{x}p_{z} + p_{y}p_{z} \\\\')
        __M_writer(' &\\quad- p_{xy}^{2} - p_{yz}^{2} - p_{xz}^{2} \\\\')
        __M_writer('J_{2} &= ')
        __M_writer(str(engr(j2_8)))
        __M_writer(' \\\\')
        __M_writer('J_{3} &= \\|\\mathbf{p}\\| = ')
        __M_writer(str(round(j3_8, 4)))
        __M_writer('\r\n\\end{split}\r\n++++\r\n\r\n* stem:[\\epsilon_{1} = ')
        __M_writer(str(engr(np.sort(np.abs(principals_8))[::-1][0])))
        __M_writer(']\r\n* stem:[\\epsilon_{2} = ')
        __M_writer(str(engr(np.sort(np.abs(principals_8))[::-1][1])))
        __M_writer(']\r\n* stem:[\\epsilon_{3} = ')
        __M_writer(str(engr(np.sort(np.abs(principals_8))[::-1][2])))
        __M_writer(']\r\n\r\nThe maximum principal strain (magnitude and direction), stem:[\\epsilon_{1} = ')
        __M_writer(str(engr(mag_8)))
        __M_writer('~\\angle~')
        __M_writer(str(engr(dir_8)))
        __M_writer("~rad].\r\nThe magnitude of the shear strain is the average of the principal strains (from *Mohr's Circle*): stem:[\\gamma_{max} = ")
        __M_writer(str(engr(t_max_8)))
        __M_writer('].\r\n\r\n\r\n\r\n[#sec-9, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 9\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.28 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.28\r\nA stem:[16~mm \\times 16~mm] square stem:[ABCD] is sketched on a plate before loading.\r\nSubsequent to loading, the square becomes the rhombus illustrated in xref:fig-9-problem_statement[].\r\nDetermine the (a) modulus of elasticity, (b) Poisson\'s Ratio, and (c) the shear modulus of elasticity.\r\n[#fig-9-problem_statement]\r\n.Adapted from cite:[uguralAdvancedMechanicsMaterials2019].\r\nimage::./9-problem_statement_220221_200917_EST.png[caption=<span class="figgynumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n\r\n\r\n\r\n[#sec-10, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 10\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.52 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.52\r\nThe distribution of stress in a structural member is given (in megapascals) by Eqs. (d) of Example 1.2 of Chapter 1 (xref:eq-10-problem_statement[]).\r\nCalculate the strains at the specified point stem:[Q(\\frac{3}{4}, \\frac{1}{4}, \\frac{1}{2})] for stem:[E = 200~GPa] and stem:[\\nu = 0.25].\r\n[stem#eq-10-problem_statement, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{split}\r\n\\sigma_{x} = -x^{3} + y^{2}, &\\quad \\tau_{xy} = 5z + 2y^{2} \\\\')
        __M_writer('\\sigma_{y} = 2x^{2} + \\frac{1}{2}y^{2}, &\\quad \\tau_{xz} = xz^{3} + x^{2}y \\\\')
        __M_writer('\\sigma_{z} = 4y^{2} - z^{3}, &\\quad \\tau_{yz} = 0\r\n\\end{split}\r\n\\end{equation}\r\n++++\r\n\r\n.Solution\r\n// solution codes\r\n')
        __M_writer('\r\n\r\nThe strain state can be found from determining the strain state at the point in the displacement field:\r\n[stem#eq-10-strain_state, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n[\\epsilon] = \\begin{bmatrix}\\epsilon_{x} = \\frac{1}{E}[\\sigma_{x} - \\nu(\\sigma_{y} + \\sigma_{z})] & \\gamma_{xy} = \\frac{\\tau_{xy}}{G} & \\gamma_{xz} = \\frac{\\tau_{xz}}{G} \\\\')
        __M_writer('0 & \\epsilon_{y} = \\frac{1}{E}[\\sigma_{y} - \\nu(\\sigma_{x} + \\sigma_{z})] & \\gamma_{yz} = \\frac{\\tau_{yz}}{G} \\\\')
        __M_writer('0 & 0 & \\epsilon_{z} = \\frac{1}{E}[\\sigma_{z} - \\nu(\\sigma_{x} + \\sigma_{y})]\\end{bmatrix}\r\n\\end{equation}\r\n++++\r\nWhen xref:eq-10-problem_statement[] and point stem:[Q] are plugged into xref:eq-10-strain_state[], this yields xref:eq-5-strain_p[] for stem:[\\mathbf{\\epsilon}(x = 0, y = 2, z = 1)].\r\n[stem#eq-10-strain_p, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n[\\epsilon] = \\begin{bmatrix}')
        __M_writer(str(engr(epsilon_10[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[0][1])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[0][2])))
        __M_writer(' \\\\')
        __M_writer(str(engr(epsilon_10[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[1][1])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[1][2])))
        __M_writer(' \\\\')
        __M_writer(str(engr(epsilon_10[2][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[2][1])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[2][2])))
        __M_writer('\\end{bmatrix}\r\n\\end{equation}\r\n++++\r\n\r\n.Answer\r\nThe strain tensor at point stem:[(0, 2, 1)], stem:[[\\epsilon\\] = \\begin{bmatrix}')
        __M_writer(str(engr(epsilon_10[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[0][1])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[0][2])))
        __M_writer(' \\\\')
        __M_writer(str(engr(epsilon_10[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[1][1])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[1][2])))
        __M_writer(' \\\\')
        __M_writer(str(engr(epsilon_10[2][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[2][1])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_10[2][2])))
        __M_writer('\\end{bmatrix}].\r\n\r\n\r\n\r\n[#sec-11, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 11\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.53 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.53\r\nAn aluminum alloy plate (stem:[E = 70~GPa], stem:[\\nu = \\frac{1}{3}]) of dimensions stem:[a = 300~mm], stem:[b = 400~mm], and thickness stem:[t = 10~mm] is subjected to biaxial stresses as shown in xref:fig-11-problem_statement[].\r\nCalculate the change in (a) the length stem:[AB] and (b) the volume of the plate.\r\n[#fig-11-problem_statement]\r\n.Adapted from cite:[uguralAdvancedMechanicsMaterials2019].\r\nimage::./11-problem_statement_220221_201834_EST.png[caption=<span class="figgynumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n\r\n.Solution\r\n// solution codes\r\n')
        __M_writer('\r\n\r\n[stem#eq-11-strain_state_form, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\begin{split}\r\n\\epsilon_{x} &= \\frac{\\sigma_{x}}{E} - \\nu\\frac{\\sigma_{y}}{E} \\\\')
        __M_writer('\\epsilon_{y} &= \\frac{\\sigma_{y}}{E} - \\nu\\frac{\\sigma_{x}}{E} \\\\')
        __M_writer('\\gamma_{xy} &= \\frac{\\tau_{xy}}{G}\r\n\\end{split}\r\n\\end{equation}\r\n++++\r\nUsing the provided stress tensor, the strain state may be found by solving xref:eq-11-strain_state_form[] which yields: stem:[[\\epsilon\\] = \\begin{bmatrix}')
        __M_writer(str(engr(epsilon_11[0][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_11[0][1])))
        __M_writer(' \\\\\\ ')
        __M_writer(str(engr(epsilon_11[1][0])))
        __M_writer(' & ')
        __M_writer(str(engr(epsilon_11[1][1])))
        __M_writer('\\end{bmatrix}].\r\nTherefore, stem:[\\Delta b = \\epsilon_{y}b = ')
        __M_writer(str(engr(delta_b_11)))
        __M_writer('~mm] and stem:[\\Delta V = (\\epsilon_{x} + \\epsilon_{y})V_{0} = ')
        __M_writer(str(engr(deltaV1_11)))
        __M_writer('~mm^{3}].\r\n\r\n\r\n\r\n[#sec-12, {counter:secs}, {counter:subs},{counter:figs}]\r\n== Problem 12\r\n:subs: 0\r\n:figs: 0\r\n\r\n> Solve Problem 2.54 from the textbook.\r\n> -- Problem Statement\r\n\r\n.Problem 2.54\r\nThe steel, rectangular parallelepiped (stem:[E = 200~GPa] and stem:[\\nu = 0.3]) shown in xref:fig-12-problem_statement[] has dimensions stem:[a = 250~mm], stem:[b = 200~mm], and stem:[c = 150~mm].\r\nIt is subjected to triaxial stresses stem:[\\sigma_{x} = -60~MPa], stem:[\\sigma_{y} = -50~MPa], and stem:[\\sigma_{z} = -40~MPa] acting on the stem:[x], stem:[y], and stem:[z] faces.\r\nDetermine (a) the changes stem:[\\Delta a], stem:[\\Delta b], and stem:[\\Delta c] in the dimensions of the block, and (b) the change stem:[\\Delta V] in the volume.\r\n[#fig-12-problem_statement]\r\n.Adapted from cite:[uguralAdvancedMechanicsMaterials2019].\r\nimage::./12-problem_statement_220221_202215_EST.png[caption=<span class="figgynumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n\r\n.Solution\r\n// solution codes\r\n')
        __M_writer('\r\n\r\nFollowing a similar procedure as in xref:sec-11[], the changes in length for each side and the final volume of the parallelepiped are: stem:[\\Delta a = ')
        __M_writer(str(round(delta_a_12, 4)))
        __M_writer('~mm], stem:[\\Delta b = ')
        __M_writer(str(round(delta_b_12, 4)))
        __M_writer('~mm], stem:[\\Delta c = ')
        __M_writer(str(round(delta_c_12, 4)))
        __M_writer('~mm], and stem:[V_{f} = [1 + (\\epsilon_{x} - 2\\nu\\epsilon_{x})\\]dxdydz = V_{0} + \\Delta V = ')
        __M_writer(str(engr(deltaV1_12)))
        __M_writer("~mm^{3}].\r\n\r\n\r\n\r\n// [appendix#sec-appendix-Figures]\r\n// == Figures\r\n\r\n\r\n\r\n[bibliography]\r\n== Bibliography\r\nbibliography::[]\r\n// end document\r\n\r\n\r\n\r\n\r\n\r\n// that's all folks")
        return ''
    finally:
        context.caller_stack._pop_frame()


"""
__M_BEGIN_METADATA
{"filename": "C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Testing\\ENGR527-727 HW4\\ENGR727_Homework4_JobyAnthonyIII.adoc", "uri": "C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Testing\\ENGR527-727 HW4\\ENGR727_Homework4_JobyAnthonyIII.adoc", "source_encoding": "utf-8", "line_map": {"16": 31, "17": 32, "18": 33, "19": 34, "20": 35, "21": 36, "22": 37, "23": 81, "24": 82, "25": 83, "26": 84, "27": 85, "28": 86, "29": 87, "30": 88, "31": 89, "32": 90, "33": 91, "34": 92, "35": 93, "36": 94, "37": 95, "38": 96, "39": 97, "40": 98, "41": 99, "42": 100, "43": 101, "44": 102, "45": 103, "46": 104, "47": 105, "48": 106, "49": 107, "50": 108, "51": 109, "52": 110, "53": 111, "54": 112, "55": 113, "56": 114, "57": 115, "58": 116, "59": 117, "60": 118, "61": 119, "62": 120, "63": 121, "64": 122, "65": 123, "66": 124, "67": 125, "68": 126, "69": 127, "70": 128, "71": 129, "72": 130, "73": 131, "74": 132, "75": 133, "76": 134, "77": 135, "78": 136, "79": 137, "80": 138, "81": 139, "82": 140, "83": 141, "84": 142, "85": 143, "86": 144, "87": 145, "88": 146, "89": 147, "90": 148, "91": 149, "92": 237, "93": 238, "94": 239, "95": 240, "96": 241, "97": 242, "98": 243, "99": 244, "100": 245, "101": 246, "102": 247, "103": 248, "104": 249, "105": 250, "106": 251, "107": 252, "108": 253, "109": 254, "110": 255, "111": 256, "112": 257, "113": 258, "114": 259, "115": 260, "116": 261, "117": 262, "118": 263, "119": 264, "120": 265, "121": 266, "122": 267, "123": 268, "124": 269, "125": 270, "126": 271, "127": 272, "128": 273, "129": 274, "130": 275, "131": 276, "132": 277, "133": 278, "134": 279, "135": 280, "136": 281, "137": 282, "138": 283, "139": 284, "140": 285, "141": 286, "142": 287, "143": 288, "144": 289, "145": 290, "146": 291, "147": 292, "148": 293, "149": 294, "150": 295, "151": 296, "152": 297, "153": 298, "154": 299, "155": 300, "156": 301, "157": 302, "158": 303, "159": 304, "160": 305, "161": 306, "162": 307, "163": 308, "164": 309, "165": 310, "166": 311, "167": 312, "168": 313, "169": 314, "170": 315, "171": 316, "172": 317, "173": 318, "174": 319, "175": 370, "176": 371, "177": 372, "178": 373, "179": 374, "180": 375, "181": 376, "182": 377, "183": 378, "184": 379, "185": 380, "186": 381, "187": 382, "188": 383, "189": 384, "190": 385, "191": 386, "192": 387, "193": 388, "194": 467, "195": 468, "196": 469, "197": 470, "198": 471, "199": 472, "200": 473, "201": 474, "202": 475, "203": 476, "204": 477, "205": 478, "206": 479, "207": 480, "208": 481, "209": 482, "210": 483, "211": 484, "212": 485, "213": 486, "214": 487, "215": 488, "216": 489, "217": 490, "218": 491, "219": 492, "220": 493, "221": 494, "222": 495, "223": 496, "224": 497, "225": 498, "226": 499, "227": 500, "228": 501, "229": 502, "230": 503, "231": 504, "232": 505, "233": 506, "234": 507, "235": 508, "236": 509, "237": 510, "238": 511, "239": 512, "240": 513, "241": 514, "242": 515, "243": 516, "244": 517, "245": 518, "246": 519, "247": 520, "248": 521, "249": 522, "250": 523, "251": 524, "252": 525, "253": 526, "254": 527, "255": 528, "256": 529, "257": 530, "258": 531, "259": 532, "260": 533, "261": 555, "262": 556, "263": 557, "264": 558, "265": 559, "266": 560, "267": 561, "268": 562, "269": 563, "270": 564, "271": 565, "272": 566, "273": 567, "274": 568, "275": 596, "276": 597, "277": 598, "278": 599, "279": 600, "280": 601, "281": 602, "282": 603, "283": 604, "284": 605, "285": 606, "286": 607, "287": 608, "288": 609, "289": 610, "290": 611, "291": 612, "292": 613, "293": 614, "294": 615, "295": 616, "296": 617, "297": 618, "298": 619, "299": 620, "300": 621, "301": 622, "302": 623, "303": 624, "304": 625, "305": 626, "306": 627, "307": 628, "308": 629, "309": 686, "310": 687, "311": 688, "312": 689, "313": 690, "314": 691, "315": 692, "316": 693, "317": 694, "318": 695, "319": 696, "320": 697, "321": 698, "322": 699, "323": 700, "324": 701, "325": 702, "326": 703, "327": 704, "328": 705, "329": 706, "330": 707, "331": 708, "332": 709, "333": 775, "334": 776, "335": 777, "336": 778, "337": 779, "338": 780, "339": 781, "340": 782, "341": 783, "342": 784, "343": 785, "344": 786, "345": 787, "346": 788, "347": 789, "348": 790, "349": 791, "350": 792, "351": 836, "352": 837, "353": 838, "354": 839, "355": 840, "356": 841, "357": 842, "358": 843, "359": 844, "360": 845, "361": 846, "362": 847, "363": 848, "364": 849, "365": 850, "366": 851, "367": 852, "368": 853, "369": 854, "370": 855, "371": 856, "372": 857, "373": 858, "374": 859, "375": 860, "376": 861, "377": 895, "378": 896, "379": 897, "380": 898, "381": 899, "382": 900, "383": 901, "384": 902, "385": 903, "386": 904, "387": 905, "388": 906, "389": 907, "390": 908, "391": 909, "392": 910, "393": 911, "394": 912, "395": 913, "396": 914, "397": 915, "398": 916, "399": 917, "400": 918, "401": 919, "402": 920, "403": 921, "404": 0, "410": 1, "411": 36, "412": 148, "413": 156, "414": 166, "415": 167, "416": 168, "417": 169, "418": 170, "419": 171, "420": 172, "421": 173, "422": 174, "423": 175, "424": 177, "425": 178, "426": 179, "427": 180, "428": 181, "429": 182, "430": 183, "431": 184, "432": 185, "433": 195, "434": 195, "435": 196, "436": 196, "437": 196, "438": 200, "439": 200, "440": 200, "441": 200, "442": 201, "443": 201, "444": 201, "445": 201, "446": 209, "447": 214, "448": 214, "449": 214, "450": 214, "451": 215, "452": 215, "453": 215, "454": 215, "455": 318, "456": 325, "457": 325, "458": 326, "459": 326, "460": 326, "461": 335, "462": 336, "463": 340, "464": 340, "465": 340, "466": 340, "467": 340, "468": 340, "469": 340, "470": 340, "471": 341, "472": 341, "473": 341, "474": 341, "475": 341, "476": 341, "477": 341, "478": 341, "479": 341, "480": 341, "481": 342, "482": 342, "483": 342, "484": 342, "485": 342, "486": 342, "487": 342, "488": 342, "489": 342, "490": 342, "491": 342, "492": 342, "493": 343, "494": 343, "495": 387, "496": 394, "497": 395, "498": 402, "499": 403, "500": 410, "501": 411, "502": 416, "503": 416, "504": 416, "505": 416, "506": 416, "507": 416, "508": 417, "509": 417, "510": 417, "511": 417, "512": 417, "513": 417, "514": 418, "515": 418, "516": 418, "517": 418, "518": 418, "519": 418, "520": 437, "521": 438, "522": 532, "523": 534, "524": 534, "525": 534, "526": 534, "527": 534, "528": 534, "529": 534, "530": 534, "531": 534, "532": 534, "533": 534, "534": 534, "535": 534, "536": 534, "537": 534, "538": 534, "539": 567, "540": 569, "541": 569, "542": 570, "543": 570, "544": 570, "545": 570, "546": 588, "547": 589, "548": 628, "549": 643, "550": 643, "551": 644, "552": 645, "553": 646, "554": 646, "555": 646, "556": 647, "557": 647, "558": 647, "559": 651, "560": 651, "561": 655, "562": 655, "563": 656, "564": 656, "565": 657, "566": 657, "567": 659, "568": 659, "569": 659, "570": 659, "571": 660, "572": 660, "573": 678, "574": 679, "575": 708, "576": 716, "577": 716, "578": 717, "579": 718, "580": 719, "581": 719, "582": 719, "583": 720, "584": 720, "585": 720, "586": 724, "587": 724, "588": 725, "589": 725, "590": 726, "591": 726, "592": 728, "593": 728, "594": 728, "595": 728, "596": 729, "597": 729, "598": 767, "599": 768, "600": 791, "601": 798, "602": 799, "603": 806, "604": 806, "605": 806, "606": 806, "607": 806, "608": 806, "609": 807, "610": 807, "611": 807, "612": 807, "613": 807, "614": 807, "615": 808, "616": 808, "617": 808, "618": 808, "619": 808, "620": 808, "621": 813, "622": 813, "623": 813, "624": 813, "625": 813, "626": 813, "627": 814, "628": 814, "629": 814, "630": 814, "631": 814, "632": 814, "633": 815, "634": 815, "635": 815, "636": 815, "637": 815, "638": 815, "639": 860, "640": 867, "641": 868, "642": 872, "643": 872, "644": 872, "645": 872, "646": 872, "647": 872, "648": 872, "649": 872, "650": 873, "651": 873, "652": 873, "653": 873, "654": 920, "655": 922, "656": 922, "657": 922, "658": 922, "659": 922, "660": 922, "661": 922, "662": 922, "668": 662}}
__M_END_METADATA
"""
