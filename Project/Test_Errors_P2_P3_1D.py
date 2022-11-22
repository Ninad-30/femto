import cubic_1d
import project_trial_modular as ptm
import numpy as np
import matplotlib.pyplot as plt
import test_fem_convergence_modular

n_elts = np.array([i*i for i in range(2, 10)])
errs_L2_ptm = []
n_quad = 3

errors_L2_p2, errors_L2_p3 = [], []

for n_elt in n_elts:
    nodes_p2, elements_p2, dbc_p2 = ptm.create_mesh_1d_uniform(n_elt)
    nodes_p3, elements_p3, dbc_p3 = cubic_1d.create_mesh_1d_uniform(n_elt)

    uh_p2 = ptm.solve_bvp(nodes_p2, elements_p2, dbc_p2, n_quad)
    uh_p3 = cubic_1d.solve_bvp(nodes_p3, elements_p3, dbc_p3, n_quad)

    ua_p2, ua_p3 = np.zeros(np.shape(uh_p2)[0] + 2), np.zeros(np.shape(uh_p3)[0] + 2)
    ua_p2[1:-1], ua_p3[1:-1] = uh_p2, uh_p3
    ua_p2, ua_p3 = ua_p2[::2], ua_p3[::3]

    x = np.linspace(0, 1, 101)
    u_ap_p2, u_ap_p3 = [], []

    for i in x:
        u_ap_p2.append(ptm.fe_interpolate_unarranged(nodes_p2, elements_p2, uh_p2, i))
        u_ap_p3.append(cubic_1d.fe_interpolate(nodes_p3, elements_p3, uh_p3, i))

    u_ex_p2 = ptm.u_exact(x)
    u_ex_p3 = cubic_1d.u_exact(x)

    error_L2_p2, error_L2_p3 = [], []
    
    for i in range(len(u_ap_p2)):
        error_L2_p2.append((u_ap_p2[i] - u_ex_p2[i])**2)
        error_L2_p3.append((u_ap_p3[i] - u_ex_p3[i])**2)
    
    errors_L2_p2.append(ptm.compute_L2_error(nodes_p2, elements_p2, ua_p2, 101))
    errors_L2_p3.append(cubic_1d.compute_L2_error(nodes_p3, elements_p3, ua_p3, 101))
plt.loglog(1/n_elts, errors_L2_p2, label = 'P2')
plt.loglog(1/n_elts, errors_L2_p3, label = 'P3')
plt.loglog(1/n_elts, (1/n_elts)**2, label = 'standard')

plt.legend()
plt.show()


