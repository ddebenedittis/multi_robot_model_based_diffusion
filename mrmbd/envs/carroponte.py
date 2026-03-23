# Symbolic crane-pendulum dynamics (affine form: dq = f(q) + g(q) * F)

import sympy as sp

# Symbolic state variables
theta, dtheta, x, dx = sp.symbols('theta dtheta x dx')
q = sp.Matrix([theta, dtheta, x, dx])

# Symbolic parameters
m1, m2, M, l, grav, r = sp.symbols('m1 m2 M l grav r')
b1, b2, F = sp.symbols('b1 b2 F')
B, C, D = sp.symbols('B C D')  # derived parameters

# Dynamics: dq = f(q) + g(q) * F
f1 = dtheta
f2 = ((-b1*dtheta)/D + ((-b2*dx - grav*l*sp.sin(theta)*B) * l*sp.cos(theta)*B) / ((C + l*sp.cos(theta)*B)*D) - (grav*l*sp.sin(theta)*B)/D)
f3 = dx
f4 = ((-b2*dx - grav*l*sp.sin(theta)*B)) / (C + l*sp.cos(theta)*B)
f = sp.Matrix([f1, f2, f3, f4])

g1 = 0
g2 = (l * sp.cos(theta) * B) / ((C + l * sp.cos(theta) * B) * D)
g3 = 0
g4 = 1 / ((C + l * sp.cos(theta) * B) * D)

g_mat = sp.Matrix([[sp.simplify(g1)], [sp.simplify(g2)], [sp.simplify(g3)], [sp.simplify(g4)]])

# Output: horizontal position of suspended mass
h = x + l * sp.sin(theta)

# JAX-compatible lambdified functions
f_fun = sp.lambdify((q, B, C, D, b1, b2, grav, l), f, modules='jax')
g_fun = sp.lambdify((q, B, C, D, l), g_mat, modules='jax')
h_fun = sp.lambdify((q, l), h, modules='jax')
