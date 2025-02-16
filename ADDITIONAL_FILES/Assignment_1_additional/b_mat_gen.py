import numpy as np
M = [20, 40, 80, 160, 320]

for m in M:
    A = np.loadtxt(f"A_{m}.txt")
    u = np.random.random(m)
    b = A @ u
    np.savetxt(f"b_{m}.txt", b)
    np.savetxt(f"u_{m}.txt", u)
    
# For 320_p
A = np.loadtxt(f"A_320_p.txt")
u = np.random.random(m)
b = A @ u
np.savetxt(f"b_320_p.txt", b)
np.savetxt(f"u_320_p.txt", u)