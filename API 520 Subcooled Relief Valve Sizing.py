# API 520 (Annex D.2.2) — Subcooled liquid at PRV inlet (two-phase flashing) sizing
# Uses CoolProp for properties (NitrousOxide). Outputs required effective discharge area A and an API 526 orifice pick.
# Note: PRVs are sized by ORIFICE AREA (API 526), not Cv. This script gives required A.

import math
from CoolProp.CoolProp import PropsSI

# User inputs
fluid = "NitrousOxide"

T_C = 20.0                     # PRV inlet liquid temperature [°C] (assumption)
set_pressure_bar_g = 80.0      # PRV set pressure [bar(g)]  (edit)
overpressure_frac = 0.10       # allowable overpressure (10% typical for many cases)
Pa_bar_a = 1.01325             # downstream total backpressure [bar(a)] (atm if venting)

mdot_required = 0.56           # required relieving mass flow [kg/s] (from your worst-case scenario)
Kd = 0.65                      # API preliminary: subcooled liquid
Kb = 1.0                       # often 1.0 if conventional + low built-up backpressure
Kc = 1.0                       # 1.0 if no rupture disk; use 0.90 if disk+PRV w/o published Kc

# Unit conversions
T_K = T_C + 273.15

bar_to_Pa = 1e5
psi_to_Pa = 6894.757293168
Pa_to_psia = 1.0 / psi_to_Pa

kgm3_to_lbft3 = 0.0624279606
Jkg_to_Btu_lb = 0.000429922614  # (Btu/lb)/(J/kg)
m3kg_to_ft3lb = 16.01846337396  # (ft^3/lb)/(m^3/kg)

# Pressures (API definitions)
# Po = relieving pressure at PRV inlet (psia): set + allowable overpressure + atm
Pset_bar_g = set_pressure_bar_g
Pset_bar_a = Pset_bar_g + 1.01325

Po_bar_a = Pset_bar_a * (1.0 + overpressure_frac)
Po_Pa = Po_bar_a * bar_to_Pa

Pa_Pa = Pa_bar_a * bar_to_Pa

# Saturation pressure at To
Ps_Pa = PropsSI("P", "T", T_K, "Q", 0, fluid)

# CoolProp properties needed for omega_s (Eq D.8)
# at inlet (Po, To): rho_lo, Cp
# at saturation (Ps, To): v_l, v_v, hvls

rho_lo = PropsSI("D", "T", T_K, "P", Po_Pa, fluid)           # kg/m^3
cp = PropsSI("Cpmass", "T", T_K, "P", Po_Pa, fluid)          # J/kg-K (same as J/kg-R with scaling handled by conversion)

rho_l_sat = PropsSI("D", "T", T_K, "Q", 0, fluid)            # kg/m^3
rho_v_sat = PropsSI("D", "T", T_K, "Q", 1, fluid)            # kg/m^3
v_l = 1.0 / rho_l_sat                                        # m^3/kg
v_v = 1.0 / rho_v_sat                                        # m^3/kg

h_l = PropsSI("Hmass", "T", T_K, "Q", 0, fluid)              # J/kg
h_v = PropsSI("Hmass", "T", T_K, "Q", 1, fluid)              # J/kg
hvls = h_v - h_l                                             # J/kg

# Convert to the US-customary units used in API Eq D.8 constant (0.185 ...)
# omega_s = [0.185 * rho_lo * Cp * To * Ps * (vvls/hvls)]^2
# where: rho_lo [lb/ft^3], Cp [Btu/lb-R], To [R], Ps [psia], vvls [ft^3/lb], hvls [Btu/lb]

rho_lo_lbft3 = rho_lo * kgm3_to_lbft3
cp_Btu_lbR = cp * Jkg_to_Btu_lb              # J/kg-K -> Btu/lb-R
To_R = (T_C * 9.0/5.0) + 491.67
Ps_psia = Ps_Pa * Pa_to_psia

vvls_ft3lb = (v_v - v_l) * m3kg_to_ft3lb
hvls_Btu_lb = hvls * Jkg_to_Btu_lb

omega_s = (0.185 * rho_lo_lbft3 * cp_Btu_lbR * To_R * Ps_psia * (vvls_ft3lb / hvls_Btu_lb)) ** 2

# Step 2: Subcooling region
# eta_st = (2 omega_s)/(1 + 2 omega_s)
# high subcooling if Ps < eta_st * Po

eta_st = (2.0 * omega_s) / (1.0 + 2.0 * omega_s)

Po_psia = (Po_Pa * Pa_to_psia)
region = "low" if (Ps_psia > eta_st * Po_psia) else "high"

# Step 3: critical vs subcritical (for HIGH subcooling region)
# critical if Ps > Pa ; subcritical(all-liquid) if Ps < Pa

Pa_psia = Pa_Pa * Pa_to_psia
if region == "high":
    criticality = "critical" if (Ps_Pa > Pa_Pa) else "subcritical_all_liquid"
else:
    criticality = "needs_Fig_D3"   # low-subcooling path requires Figure D.3 (ηc)

# Step 4: mass flux G
# In SI, API Eq D.11 is equivalent to: G = sqrt(2 * rho_lo * (Po - P))  [kg/s/m^2]
# For critical: P = Ps ; for subcritical(all-liquid): P = Pa

if region != "high":
    raise SystemExit("Low-subcooling region detected. Implement Figure D.3 (ηc) path or re-check inputs (often high-subcooling for strongly subcooled liquids).")

P_use = Ps_Pa if criticality == "critical" else Pa_Pa
dP = Po_Pa - P_use
if dP <= 0:
    raise SystemExit("Po <= P_use (Ps or Pa). Check pressures / temperature / scenario.")

G = math.sqrt(2.0 * rho_lo * dP)   # kg/s/m^2

# Step 5: required effective discharge area
# A = mdot / (Kd * Kb * Kc * G)
A_m2 = mdot_required / (Kd * Kb * Kc * G)
A_in2 = A_m2 * 1550.0031000062

# API 526 orifice areas (in^2) — partial standard set
api526 = {
    "D": 0.110, "E": 0.196, "F": 0.307, "G": 0.503, "H": 0.785,
    "J": 1.287, "K": 1.838, "L": 2.853, "M": 3.600, "N": 4.340,
    "P": 6.380, "Q": 11.050, "R": 16.000, "T": 26.000
}
pick = None
for letter, area in sorted(api526.items(), key=lambda x: x[1]):
    if area >= A_in2:
        pick = letter
        break


# Sizing results
print("API 520 D.2.2 Subcooled Liquid PRV Sizing")
print(f"Fluid: {fluid}")
print(f"T_inlet = {T_C:.2f} °C  ({To_R:.2f} R)")
print(f"Pset = {Pset_bar_g:.2f} bar(g)  | Overpressure = {overpressure_frac*100:.1f}%")
print(f"Po (relieving inlet) = {Po_bar_a:.3f} bar(a)  = {Po_psia:.2f} psia")
print(f"Pa (backpressure)    = {Pa_bar_a:.3f} bar(a)  = {Pa_psia:.2f} psia")
print(f"Psat(To)             = {(Ps_Pa/bar_to_Pa):.3f} bar(a)  = {Ps_psia:.2f} psia")
print()
print(f"rho_lo(Po,To) = {rho_lo:.2f} kg/m^3  ({rho_lo_lbft3:.2f} lb/ft^3)")
print(f"Cp(Po,To)     = {cp:.1f} J/kg-K  ({cp_Btu_lbR:.4f} Btu/lb-R)")
print(f"hvls(To)      = {hvls/1000:.2f} kJ/kg  ({hvls_Btu_lb:.2f} Btu/lb)")
print(f"vvls(To)      = {(v_v-v_l):.6e} m^3/kg  ({vvls_ft3lb:.6e} ft^3/lb)")
print()
print(f"omega_s = {omega_s:.4f}")
print(f"eta_st  = {eta_st:.4f}")
print(f"Region  = {region} subcooling")
print(f"Flow    = {criticality}")
print(f"Mass flux G = {G:.2f} kg/s/m^2")
print()
print(f"mdot_required = {mdot_required:.6f} kg/s")
print(f"A_required = {A_m2:.6e} m^2  = {A_in2:.6f} in^2")
print(f"API 526 orifice pick: {pick if pick else 'Larger than T / custom'}")