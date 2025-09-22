import numpy as np
import matplotlib.pyplot as plt


MASS = 1
SPRING = 20

x0 = 1
v0 = 10
DT = 0.001


def force(pos, k=SPRING):
    return -k * pos

def energy(pos, vel, k=SPRING, m=MASS):
    potential = 1 / 2 * k * pos**2
    kinetic = 1 / 2 * m * vel**2
    total = potential + kinetic
    return potential, kinetic, total

def explicit(pos, vel, k=SPRING, m=MASS, dt=DT):
    a = force(pos, k) / m
    vel_new = vel + a * dt
    pos_new = pos + vel * dt
    return pos_new, vel_new

def symplectic(pos, vel, k=SPRING, m=MASS, dt=DT):
    a = force(pos, k) / m
    vel_new = vel + a * dt
    pos_new = pos + vel_new * dt
    return pos_new, vel_new

def runge_kutte(pos, vel, k=SPRING, m=MASS, dt=DT):
    a = force(pos, k) / m
    j = -k / m * vel
    vel_new = vel + a * dt + 1/2 * j * dt**2
    pos_new = pos + vel_new * dt + 1/2 * a * dt**2
    return pos_new, vel_new

def simulate(method, k=SPRING, m=MASS, dt=DT):
    pos = x0
    vel = v0
    t = 0
    pe, ke, e = energy(pos, vel, k, m)

    t_hist = [0] 
    x_hist = [pos]
    v_hist = [vel]
    pe_hist, ke_hist, e_hist = [pe], [ke], [e]

    while t < 50: 
        pos, vel = method(pos, vel, k, m, dt)
        pe, ke, e = energy(pos, vel, k, m)
        t += dt

        t_hist.append(t)
        x_hist.append(pos)
        v_hist.append(vel)
        pe_hist.append(pe)
        ke_hist.append(ke)
        e_hist.append(e)

    return [np.array(t_hist), np.array(x_hist), np.array(v_hist),
            np.array(pe_hist), np.array(ke_hist), np.array(e_hist)]



if __name__ == "__main__":
    res_exp = simulate(explicit, SPRING, MASS, DT)
    res_sym = simulate(symplectic, SPRING, MASS, DT)
    res_run = simulate(runge_kutte, SPRING, MASS, DT)

    plt.figure()
    plt.plot(res_exp[0], res_exp[-1], label="explicit")
    plt.plot(res_sym[0], res_sym[-1], label="symplectic")
    plt.plot(res_run[0], res_run[-1], label="runge-kutte")
    plt.xlabel("time (s)")
    plt.ylabel("energy (J)")
    plt.title("Energy vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()
