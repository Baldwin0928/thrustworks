import math

# Thrust
F_lbf = 1500.0              # thrust [lbf]

# Chamber / exit pressure
Pc_psi = 300.0              # chamber pressure [psi]
Pe_psi = 14.7               # nozzle exit pressure [psi]

# Performance estimates from CEA or literature
cstar_m_s = 1779.870546     # characteristic velocity [m/s]
Cf = 1.4059                 # thrust coefficient [-]
Isp_s = 255.1656274         # specific impulse [s]
gamma = 1.1692              # throat/exit effective gas gamma [-]

# Mixture ratio
OF = 2.0                    # oxidizer-to-fuel ratio [-]

# Chamber geometry
Lstar_in = 50.0             # characteristic length, L* [in], check H&H
epsilon_c = 5.0             # contraction ratio, Ac / At [-], check H&H

# Nozzle geometry, assuming a conical nozzle
theta_conv_deg = 45.0       # converging half-angle [deg], typical value
theta_div_deg = 15.0        # conical divergent half-angle [deg]

# CONSTANTS
g0 = 9.80665                # gravity [m/s^2]
lbf_to_N = 4.4482216152605
psi_to_Pa = 6894.757293168
m2_to_in2 = 1550.0031
m_to_in = 39.3700787
in_to_m = 0.0254

# Throat area

F_N = F_lbf * lbf_to_N
Pc_Pa = Pc_psi * psi_to_Pa

# Method 1: Using thrust coefficient
At_from_Cf_m2 = F_N / (Cf * Pc_Pa)

# Method 2: Using c* and Isp
mdot_total = F_N / (Isp_s * g0)
At_from_cstar_m2 = mdot_total * cstar_m_s / Pc_Pa

# Choose throat area method for geometry
# For consistency with thrust sizing, use Cf method by default
At_m2 = At_from_Cf_m2
At_in2 = At_m2 * m2_to_in2

# Calculing ox and fuel mass flow rates
mdot_ox = mdot_total * OF / (1.0 + OF)
mdot_fuel = mdot_total / (1.0 + OF)

# Throat diameter
Dt_m = math.sqrt(4.0 * At_m2 / math.pi)
Dt_in = Dt_m * m_to_in

# Chamber + Converging section calculations
theta_conv_rad = math.radians(theta_conv_deg)
cot_theta_conv = 1.0 / math.tan(theta_conv_rad)

# Chamber area
Ac_in2 = epsilon_c * At_in2

# Chamber diameter
Dc_in = math.sqrt(4.0 * Ac_in2 / math.pi)
Dc_m = Dc_in * in_to_m

# Radii based on areas
rt_in = math.sqrt(At_in2 / math.pi)
rc_in = math.sqrt(Ac_in2 / math.pi)

# Converging section axial length
L_conv_in = (rc_in - rt_in) * cot_theta_conv
L_conv_m = L_conv_in * in_to_m

# Converging-section volume contribution divided by At
conv_volume_term_in = (
    (1.0 / 3.0)
    * math.sqrt(At_in2 / math.pi)
    * cot_theta_conv
    * (epsilon_c ** 1.5 - 1.0)
)

# Cylindrical chamber length from L*
Lc_in = (Lstar_in - conv_volume_term_in) / epsilon_c
Lc_m = Lc_in * in_to_m

# Injector face to throat length
L_total_to_throat_in = Lc_in + L_conv_in
L_total_to_throat_m = L_total_to_throat_in * in_to_m

# Chamber volume check
Vc_required_in3 = Lstar_in * At_in2
Vc_geometry_in3 = At_in2 * (
    Lc_in * epsilon_c + conv_volume_term_in
)
volume_error_in3 = Vc_geometry_in3 - Vc_required_in3

# Expansion Ratio and Diverging Section Calculations
theta_div_rad = math.radians(theta_div_deg)

# Exit Mach number from isentropic pressure ratio
Me = math.sqrt(
    (2.0 / (gamma - 1.0))
    * ((Pc_psi / Pe_psi) ** ((gamma - 1.0) / gamma) - 1.0)
)

# Area-Mach relation
epsilon = (
    (1.0 / Me)
    * (
        (2.0 / (gamma + 1.0))
        * (1.0 + ((gamma - 1.0) / 2.0) * Me**2)
    ) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
)

# Exit area
Ae_in2 = epsilon * At_in2
Ae_m2 = Ae_in2 / m2_to_in2

# Exit diameter
De_in = math.sqrt(4.0 * Ae_in2 / math.pi)
De_m = De_in * in_to_m

# Divergent section length
re_in = De_in / 2.0
L_div_in = (re_in - rt_in) / math.tan(theta_div_rad)
L_div_m = L_div_in * in_to_m

# Total thrust chamber length
L_total_in = Lc_in + L_conv_in + L_div_in
L_total_m = L_total_in * in_to_m

print("TCA Internal Dimensions")

print("\nInputs")
print("-------")
print(f"Thrust:                         {F_lbf:.3f} lbf = {F_N:.3f} N")
print(f"Chamber pressure, Pc:           {Pc_psi:.3f} psi = {Pc_Pa:.3f} Pa")
print(f"Exit pressure, Pe:              {Pe_psi:.3f} psi")
print(f"c*:                             {cstar_m_s:.3f} m/s")
print(f"Cf:                             {Cf:.6f}")
print(f"Isp:                            {Isp_s:.6f} s")
print(f"Gamma:                          {gamma:.6f}")
print(f"O/F:                            {OF:.3f}")
print(f"L*:                             {Lstar_in:.3f} in")
print(f"Contraction ratio, Ac/At:       {epsilon_c:.3f}")
print(f"Converging half-angle:          {theta_conv_deg:.3f} deg")
print(f"Diverging half-angle:           {theta_div_deg:.3f} deg")

print("\nMASS FLOW")
print("-------")
print(f"Total mdot:                     {mdot_total:.6f} kg/s")
print(f"Oxidizer mdot:                  {mdot_ox:.6f} kg/s")
print(f"Fuel mdot:                      {mdot_fuel:.6f} kg/s")

print("\nTHROAT SIZING")
print("-------")
print("Using Cf method for geometry sizing.")
print(f"Throat area, At:                {At_in2:.6f} in^2")
print(f"Throat area, At:                {At_m2:.8e} m^2")
print(f"Throat diameter, Dt:            {Dt_in:.6f} in")
print(f"Throat diameter, Dt:            {Dt_m:.6f} m")

print("\nCHAMBER GEOMETRY")
print("-------")
print(f"Chamber area, Ac:               {Ac_in2:.6f} in^2")
print(f"Chamber diameter, Dc:           {Dc_in:.6f} in")
print(f"Chamber diameter, Dc:           {Dc_m:.6f} m")
print(f"Cylindrical chamber length, Lc: {Lc_in:.6f} in")
print(f"Cylindrical chamber length, Lc: {Lc_m:.6f} m")

print("\nCONVERGING SECTION")
print("-------")
print(f"Converging length, L_conv:      {L_conv_in:.6f} in")
print(f"Converging length, L_conv:      {L_conv_m:.6f} m")
print(f"Injector face to throat:        {L_total_to_throat_in:.6f} in")
print(f"Injector face to throat:        {L_total_to_throat_m:.6f} m")

print("\nCHAMBER VOLUME CHECK")
print("-------")
print(f"Required volume, Vc = L*At:     {Vc_required_in3:.6f} in^3")
print(f"Geometry volume:                {Vc_geometry_in3:.6f} in^3")
print(f"Volume error:                   {volume_error_in3:.6e} in^3")

print("\nNOZZLE EXPANSION")
print("-------")
print(f"Exit Mach number, Me:           {Me:.6f}")
print(f"Expansion ratio, Ae/At:         {epsilon:.6f}")
print(f"Exit area, Ae:                  {Ae_in2:.6f} in^2")
print(f"Exit area, Ae:                  {Ae_m2:.8e} m^2")
print(f"Exit diameter, De:              {De_in:.6f} in")
print(f"Exit diameter, De:              {De_m:.6f} m")

print("\nDIVERGING SECTION")
print("-------")
print(f"Diverging length, L_div:        {L_div_in:.6f} in")
print(f"Diverging length, L_div:        {L_div_m:.6f} m")

print("\nFINAL LENGTH SUMMARY")
print("-------")
print(f"Cylindrical chamber length:     {Lc_in:.6f} in")
print(f"Converging section length:      {L_conv_in:.6f} in")
print(f"Diverging section length:       {L_div_in:.6f} in")
print(f"Injector face to throat:        {L_total_to_throat_in:.6f} in")
print(f"Injector face to nozzle exit:   {L_total_in:.6f} in")
print(f"Injector face to nozzle exit:   {L_total_m:.6f} m")

print("\nDIAMETER SUMMARY")
print("-------")
print(f"Chamber diameter, Dc:           {Dc_in:.6f} in")
print(f"Throat diameter, Dt:            {Dt_in:.6f} in")
print(f"Exit diameter, De:              {De_in:.6f} in")

print("\n-------")

# Warnings

if Lc_in <= 0:
    print("\nWARNING:")
    print("The calculated cylindrical chamber length is zero or negative.")
    print("Your chosen L*, contraction ratio, throat area, and converging")
    print("angle combination is not physically reasonable.")
    print("Try increasing L*, decreasing contraction ratio, or increasing")
    print("the converging half-angle.")

if Me <= 1.0:
    print("\nWARNING:")
    print("The calculated exit Mach number is not supersonic.")
    print("Check Pc, Pe, and gamma.")

if theta_conv_deg <= 0 or theta_conv_deg >= 90:
    print("\nWARNING:")
    print("Converging half-angle should usually be between 0 and 90 degrees.")

if theta_div_deg <= 0 or theta_div_deg >= 45:
    print("\nWARNING:")
    print("Diverging half-angle is unusual.")
    print("Typical conical nozzle divergent half-angles are often around 12 to 18 degrees.")