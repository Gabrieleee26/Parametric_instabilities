# Parametric Instabilities: A Study of SRS and SBS in Shock Ignition Conditions

This repository contains a study evaluating the competition between **Stimulated Raman Scattering (SRS)** and **Stimulated Brillouin Scattering (SBS)**, focusing on their role in laser energy absorption efficiency under **shock ignition** conditions.

The goal of this study is to understand how these two parametric instabilities influence laser-plasma interactions in an expanding, underdense, inhomogeneous plasma composed of plastic (CH). We simulate four representative cases based on the interaction conditions.

---

## Representative Cases

We simulate the following four cases to explore different physical conditions:

1. **Reference Case**:  
   Interaction of a monochromatic laser pulse with a **collisionless** plasma, where the fluid velocity is zero, and the plasma consists of a single ion species. The effective charge is $Z_{\text{eff}} = 3.5$ and the effective mass is $A_{\text{eff}} = 6.5$, corresponding to an equimolar mixture of hydrogen and carbon.

2. **Collision Case**:  
   The reference case with **electron-ion collisions** switched on, adding realistic physical interactions between particles in the plasma.

3. **Expanding Plasma Case**:  
   The reference case with an **expanding plasma**, where the fluid velocity increases linearly with position: $u(x) = u_s \left((x/l_u) - 1 \right)$, with $l_u = 300 \lambda_0$, $u_s = 0.75 \,\mu\text{m}/\text{ps}$.
   This case models the expansion of the plasma, a typical feature in shock ignition scenarios.

4. **Expanding Plasma with Collisions**:  
   The third case with **electron-ion collisions** included. This case models both plasma expansion and the more realistic collision effects.

---

## Input Parameters

We simulate an intense laser pulse incident normally on an expanding, underdense, inhomogeneous plasma composed of plastic (CH), with parameters relevant to the shock ignition scenario. The following input parameters are used:

- **Plasma density profile**:
  $n_e/n_{\text{cr}} = 0.05 + 0.23 (x/l_n)$,
  where $l_n = 286 \lambda_0$, and $\lambda_0 = 0.351 \,\mu\text{m}$ is the laser wavelength.

- **Initial temperatures**:
  - Electron temperature: $T_e = 3.4 \, \text{keV}$
  - Ion temperature: $T_i = 1.0 \, \text{keV}$

- **Laser pulse intensity**:
  $I_0 = 1/2 c \epsilon_0 E_0^2 = 6 \times 10^{15} \, \text{W/cm}^2$
  The laser intensity remains constant for the simulation time of **8 ps**, corresponding to about **10,000 laser periods**, after a linear ramp during the first ten laser periods.

- **Simulation box**:
  - Length: $314 \lambda_0$
  - Vacuum margins: $14 \lambda_0$ at both the front and rear sides of the box to allow free plasma expansion.
