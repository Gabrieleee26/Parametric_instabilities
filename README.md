# Parametric Instabilities

In this study, we evaluate the competition between Stimulated Raman Scattering (SRS) and Stimulated Brillouin Scattering (SBS) and their role in the efficiency of laser energy absorption under shock ignition conditions.

We consider the following four representative cases:

- **Reference case**: Interaction of a monochromatic laser pulse with a collisionless plasma with zero fluid velocity and a single ion species. The effective charge is \(Z_{\text{eff}}=3.5\) and the effective mass is \(A_{\text{eff}}=6.5\), corresponding to an equimolar mixture of hydrogen and carbon.
- **Collision case**: The reference case with electron-ion collisions switched on.
- **Expanding plasma case**: The reference case with an expanding plasma where the fluid velocity increases linearly with position:  
  \[
  u(x) = u_s \left( \frac{x}{l_u} - 1 \right), \quad \text{with } l_u = 300 \lambda_0, \quad u_s = 0.75 \,\mu\text{m}/\text{ps}.
  \]
- **Expanding plasma with collisions**: The third case with electron-ion collisions included.

## Input Parameters

We study an intense laser pulse incident normally on an expanding, underdense, inhomogeneous plasma composed of plastic (CH) with parameters relevant to the shock ignition scenario. The plasma has a linear density profile:

\[
\frac{n_e}{n_{cr}} = 0.05 + 0.23 \frac{x}{l_n},
\]

over a length \( l_n = 286 \lambda_0 \), where the laser wavelength is \( \lambda_0 = 0.351 \,\mu\text{m} \).

The initial electron temperature is \( T_e = 3.4 \) keV, and the ion temperature is \( T_i = 1.0 \) keV, both assumed to be homogeneous in space.

The laser pulse intensity is given by:

\[
I_0 = \frac{1}{2} c \epsilon_0 E_0^2 = 6 \times 10^{15} \, W/\text{cm}^2,
\]

which remains constant throughout the simulation time of 8 ps, approximately corresponding to \( 10^4 \) laser periods, after a linear ramp during the first ten laser periods.

The simulation box length is \( 314 \lambda_0 \) with \( 14 \lambda_0 \) vacuum margins at the front and rear sides of the box to enable free plasma expansion.
