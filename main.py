import numpy as np
import matplotlib.pyplot as plt


MASS = 1  # kg
SPRING = 20  # kg/s^2

x0 = 1  # m
v0 = 5  # m/s
DT = 0.001  # s


def force(pos, k=SPRING):
    return -k * pos


def energy(pos, vel, k=SPRING, m=MASS, dt=DT):
    potential = 1 / 2 * k * pos**2
    kinetic = 1 / 2 * m * vel**2
    total = potential + kinetic
    return total


def energy_cor(pos, vel, k=SPRING, m=MASS, dt=DT):
    """Add the energy correction term"""
    return energy(pos, vel, k, m) - 1 / 2 * k * dt * vel * pos


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
    vel_new = vel + a * dt + 1 / 2 * j * dt**2
    pos_new = pos + vel_new * dt + 1 / 2 * a * dt**2
    return pos_new, vel_new


def simulate(method, which_energy, k=SPRING, m=MASS, dt=DT):
    pos = x0
    vel = v0
    t = 0
    e = which_energy(pos, vel, k, m)

    t_hist = [0]
    x_hist = [pos]
    v_hist = [vel]
    e_hist = [e]

    while t < 50:
        pos, vel = method(pos, vel, k, m, dt)
        e = which_energy(pos, vel, k, m)
        t += dt

        t_hist.append(t)
        x_hist.append(pos)
        v_hist.append(vel)
        e_hist.append(e)

    return [np.array(t_hist), np.array(x_hist), np.array(v_hist), np.array(e_hist)]


if __name__ == "__main__":
    # Simulate with the theoretical Hamiltonian
    res_exp = simulate(explicit, energy, SPRING, MASS, DT)
    res_sym = simulate(symplectic, energy, SPRING, MASS, DT)
    res_run = simulate(runge_kutte, energy, SPRING, MASS, DT)

    # Simulate with the Hamiltonian with correction
    res_exp_cor = simulate(explicit, energy_cor, SPRING, MASS, DT)
    res_sym_cor = simulate(symplectic, energy_cor, SPRING, MASS, DT)
    res_run_cor = simulate(runge_kutte, energy_cor, SPRING, MASS, DT)

    plt.figure()
    plt.plot(res_exp[0], res_exp[-1], label="explicit")
    plt.plot(res_sym[0], res_sym[-1], label="symplectic")
    plt.plot(res_run[0], res_run[-1], label="runge-kutte")
    plt.xlabel("time (s)")
    plt.ylabel("energy (J)")
    plt.title("Energy vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_time.png")
    plt.show()

    plt.figure()
    plt.plot(res_exp_cor[0], res_exp_cor[-1], label="explicit")
    plt.plot(res_sym_cor[0], res_sym_cor[-1], label="symplectic")
    plt.plot(res_run_cor[0], res_run_cor[-1], label="runge-kutte")
    plt.xlabel("time (s)")
    plt.ylabel("energy (J)")
    plt.title("Energy vs Time with Correction")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_time_cor.png")
    plt.show()

    plt.figure()
    plt.plot(res_exp[0], res_exp[1], label="explicit")
    plt.plot(res_sym[0], res_sym[1], label="symplectic")
    plt.plot(res_run[0], res_run[1], label="runge-kutte")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Position vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("position_time.png")
    plt.show()
