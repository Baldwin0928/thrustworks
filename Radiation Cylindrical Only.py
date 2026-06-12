from pathlib import Path
from datetime import datetime
import csv
import math
import os
import re
import subprocess

from rp1_rocketprops import (
    rp1_density,
    rp1_fallback_summary,
    rp1_saturation_props,
    rp1_surface_tension,
    rp1_vapor_density_at_saturation,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CEA_EXE = Path(os.environ.get("CEA_EXE", SCRIPT_DIR / "cea.exe"))
CEA_DATA_DIR = Path(os.environ.get("CEA_DATA_DIR", SCRIPT_DIR / "data"))
CEA_RUN_DIR = Path(os.environ.get("CEA_RUN_DIR", SCRIPT_DIR))

# Chamber conditions and geometry
PC_PSI = 300.0
OF_RATIO = 2.0
LOX_TEMP_K = 90.17
RP1_TEMP_K = 298.15
TWALL_K = 1200.0

DT_IN = 2.747
DC_IN = DT_IN * math.sqrt(5.0)

# reference chamber length from injector to throat
LCHAMBER_TOTAL_TO_THROAT_IN = 9.06775

# Grissom infinite-cylinder simplification.
USE_LEFF_095D = True
LEFF_DIAMETER_SOURCE = "chamber"  # "chamber" or "throat"

# Optional Egelstaff reflective-wall correction: Leff_corrected = Leff * Aw^-0.85.
APPLY_REFLECTIVE_WALL_CORRECTION = False
WALL_ABSORPTIVITY = 1.0

OUTPUT_CSV = Path(__file__).with_name("grisson_radiation_output.csv")
SIGMA = 5.67e-8
PSI_PER_ATM = 14.6959487755

# Optional liquid-film burnout warning. RP-1 properties are pulled from
# RocketProps at this pressure.
BURNOUT_CORRELATION = "Katto-Ishii"
BURNOUT_PROPERTY_PRESSURE_PA = PC_PSI * 6894.757
BURNOUT_HEATED_LENGTH_M = 0.05
FILM_AVERAGE_VELOCITY_M_S = 1.5
FILM_THICKNESS_M = 0.0005
LIQUID_ABSORPTIVITY_1_M = 0.0


def cea_float(token, default=math.nan):
    try:
        token = str(token).strip().replace("D", "E")
        if "E" not in token.upper():
            match = re.match(r"^([-+]?\d*\.?\d+)([-+]\d+)$", token)
            if match:
                token = match.group(1) + "E" + match.group(2)
        return float(token)
    except Exception:
        return default


def build_cea_input_text(pc_psi, of_ratio):
    pc_bar = pc_psi * 0.0689475729
    return f"""problem rocket equilibrium
  p(bar)={pc_bar}
  o/f={of_ratio}
reactants
  fuel=RP-1 wt%=100 t(k)={RP1_TEMP_K}
  oxid=O2(L) wt%=100 t(k)={LOX_TEMP_K}
output siunits
end
"""


def run_cea(pc_psi, of_ratio):
    run_dir = CEA_RUN_DIR / "CEA_Results" / datetime.now().strftime("grisson_%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    case_name = f"grisson_pc_{pc_psi:.1f}_of_{of_ratio:.3f}".replace(".", "p")
    base = run_dir / case_name
    base.with_suffix(".inp").write_text(build_cea_input_text(pc_psi, of_ratio), encoding="utf-8")

    env = os.environ.copy()
    env["CEA_DATA_DIR"] = str(CEA_DATA_DIR)
    subprocess.run([str(CEA_EXE), str(base)], check=True, cwd=str(run_dir), env=env)
    return base.with_suffix(".out").read_text(encoding="utf-8", errors="ignore"), run_dir


def parse_property_row(out_text, row_name):
    number_pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+|[-+]\d+)?"
    row_name_upper = row_name.upper()
    for line in out_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith(row_name_upper):
            return [cea_float(x) for x in re.findall(number_pattern, stripped[len(row_name):])]
    return []


def parse_species_mole_fraction(out_text, species):
    number_pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+|[-+]\d+)?"
    species_upper = species.upper()
    in_mole_fraction_block = False
    for line in out_text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("MOLE FRACTIONS"):
            in_mole_fraction_block = True
            continue
        if in_mole_fraction_block and (
            upper.startswith("THERMODYNAMIC")
            or upper.startswith("PRODUCTS WHICH")
            or upper.startswith("PERFORMANCE")
        ):
            break
        if in_mole_fraction_block and upper.startswith(species_upper):
            nums = [cea_float(x) for x in re.findall(number_pattern, stripped[len(species):])]
            return nums[0] if nums else math.nan
    return math.nan


def parse_cea_chamber_state(out_text):
    temps = parse_property_row(out_text, "T, K")
    return {
        "Tg_K": temps[0] if temps else math.nan,
        "N_H2O": parse_species_mole_fraction(out_text, "H2O"),
        "N_CO2": parse_species_mole_fraction(out_text, "CO2"),
    }


def interpolate_coefficients(T_K, species):
    tables = {
        "H2O": {
            "T": [1000.0, 2000.0, 3000.0],
            "c": [0.165, 0.90, 2.05],
            "n": [0.45, 0.65, 0.61],
        },
        "CO2": {
            "T": [1000.0, 1500.0, 2000.0],
            "c": [0.05, 0.075, 0.15],
            "n": [0.6, 0.6, 0.6],
        },
    }
    data = tables[species]
    T_values = data["T"]

    if T_K <= T_values[0]:
        i0, i1 = 0, 1
    elif T_K >= T_values[-1]:
        i0, i1 = len(T_values) - 2, len(T_values) - 1
    else:
        for i in range(len(T_values) - 1):
            if T_values[i] <= T_K <= T_values[i + 1]:
                i0, i1 = i, i + 1
                break

    frac = (T_K - T_values[i0]) / (T_values[i1] - T_values[i0])
    c = data["c"][i0] + frac * (data["c"][i1] - data["c"][i0])
    n = data["n"][i0] + frac * (data["n"][i1] - data["n"][i0])
    return c, n


def one_atm_emittance(rho_opt_atm_m, T_K, species):
    eps_inf = {"H2O": 0.825, "CO2": 0.231}[species]
    if rho_opt_atm_m <= 0.0:
        return 0.0
    c, n = interpolate_coefficients(T_K, species)
    return eps_inf * (1.0 + (rho_opt_atm_m / c) ** (-n)) ** (-1.0 / n)


def h2o_pressure_correction(P_atm, p_h2o_atm, N_h2o):
    C1 = 0.26 + 0.74 * math.exp(-2.5 * p_h2o_atm)
    C2 = 0.75 + 0.31 * math.exp(-10.0 * p_h2o_atm)
    return 1.0 + C1 * (1.0 - math.exp(((1.0 - P_atm) * (1.0 + N_h2o)) / C2))


def co2_pressure_correction(P_atm, p_co2_atm):
    if p_co2_atm <= 0.0 or P_atm <= 0.0:
        return 1.0
    m = 100.0 * p_co2_atm
    pressure_term = 1.0 + (2.0 * math.log10(P_atm)) ** (-m)
    log10_kp = 0.036 * p_co2_atm ** (-0.45) * pressure_term ** (-1.0 / m)
    return 10.0 ** log10_kp


def spectral_overlap_correction(rho_opt_total_atm_m, N_h2o, N_co2):
    if rho_opt_total_atm_m <= 0.0 or N_h2o <= 0.0 or N_co2 <= 0.0:
        return 0.0
    n = 5.5 * (1.0 + (1.09 * rho_opt_total_atm_m) ** (-3.88)) ** (-1.0 / 3.88)
    kx = 1.0 - abs((2.0 * N_h2o / (N_h2o + N_co2)) - 1.0) ** n
    return (
        0.0551
        * kx
        * (1.0 - math.exp(-4.0 * rho_opt_total_atm_m))
        * (1.0 - math.exp(-12.5 * rho_opt_total_atm_m))
    )


def calculate_grissom_radiation(Tg_K, Twall_K, pc_psi, N_h2o, N_co2, leff_m):
    P_atm = pc_psi / PSI_PER_ATM
    p_h2o_atm = N_h2o * P_atm
    p_co2_atm = N_co2 * P_atm

    rho_h2o = p_h2o_atm * leff_m
    rho_co2 = p_co2_atm * leff_m
    rho_total = (p_h2o_atm + p_co2_atm) * leff_m

    eps_h2o_1atm = one_atm_emittance(rho_h2o, Tg_K, "H2O")
    eps_co2_1atm = one_atm_emittance(rho_co2, Tg_K, "CO2")

    kp_h2o = h2o_pressure_correction(P_atm, p_h2o_atm, N_h2o)
    kp_co2 = co2_pressure_correction(P_atm, p_co2_atm)

    eps_h2o = eps_h2o_1atm * kp_h2o
    eps_co2 = eps_co2_1atm * kp_co2
    delta_eps = spectral_overlap_correction(rho_total, N_h2o, N_co2) if Tg_K > 1200.0 else 0.0
    eps_gas = max(0.0, min(1.0, eps_h2o + eps_co2 - delta_eps))

    q_rad_W_m2 = SIGMA * eps_gas * (Tg_K**4 - Twall_K**4)

    return {
        "Pc_psi": pc_psi,
        "Pc_atm": P_atm,
        "OF_ratio": OF_RATIO,
        "Tg_K": Tg_K,
        "Twall_K": Twall_K,
        "N_H2O": N_h2o,
        "N_CO2": N_co2,
        "p_H2O_atm": p_h2o_atm,
        "p_CO2_atm": p_co2_atm,
        "Leff_m": leff_m,
        "rho_H2O_atm_m": rho_h2o,
        "rho_CO2_atm_m": rho_co2,
        "rho_total_atm_m": rho_total,
        "epsilon_H2O_1atm": eps_h2o_1atm,
        "epsilon_CO2_1atm": eps_co2_1atm,
        "Kp_H2O": kp_h2o,
        "Kp_CO2": kp_co2,
        "epsilon_H2O": eps_h2o,
        "epsilon_CO2": eps_co2,
        "delta_epsilon": delta_eps,
        "epsilon_gas": eps_gas,
        "q_rad_W_m2": q_rad_W_m2,
        "q_rad_MW_m2": q_rad_W_m2 / 1.0e6,
    }


def calculate_liquid_film_burnout_check(q_rad_W_m2):
    """
    Order-of-magnitude film burnout warning from the paper's Katto-Ishii form:

        q_bo / (rho_v * lambda * U)
            = C * (rho_l / rho_v)^n1 * (sigma / (rho_l * L * U^2))^n2

    This is used as a warning to see if film detaches, but this burnout check
    relies on many variables which you will not be sure of. So, don't really
    trust it is what I am saying - though it is kept here in case you do have
    the needed values to make a somewhat correct guess.
    """
    coefficients = {
        "Monde-Katto": {"C": 0.0591, "n1": 0.725, "n2": 0.333},
        "Katto-Ishii": {"C": 0.0164, "n1": 0.867, "n2": 0.333},
        "Mudawwar": {"C": 0.0881, "n1": 0.867, "n2": 0.432},
    }
    coeff = coefficients[BURNOUT_CORRELATION]
    T_sat_K, h_fg_J_kg, _ = rp1_saturation_props(BURNOUT_PROPERTY_PRESSURE_PA)
    rho_l_kg_m3 = rp1_density(T_sat_K, BURNOUT_PROPERTY_PRESSURE_PA)
    rho_v_kg_m3 = rp1_vapor_density_at_saturation(T_sat_K)
    surface_tension_N_m = rp1_surface_tension(T_sat_K)

    density_ratio_term = (rho_l_kg_m3 / rho_v_kg_m3) ** coeff["n1"]
    surface_tension_term = (
        surface_tension_N_m
        / (rho_l_kg_m3 * BURNOUT_HEATED_LENGTH_M * FILM_AVERAGE_VELOCITY_M_S**2)
    ) ** coeff["n2"]
    q_burnout_W_m2 = (
        rho_v_kg_m3
        * h_fg_J_kg
        * FILM_AVERAGE_VELOCITY_M_S
        * coeff["C"]
        * density_ratio_term
        * surface_tension_term
    )

    transmitted_fraction = math.exp(-LIQUID_ABSORPTIVITY_1_M * FILM_THICKNESS_M)
    q_rad_transmitted_W_m2 = q_rad_W_m2 * transmitted_fraction

    return {
        "burnout_correlation": BURNOUT_CORRELATION,
        "burnout_C": coeff["C"],
        "burnout_n1": coeff["n1"],
        "burnout_n2": coeff["n2"],
        "burnout_property_pressure_Pa": BURNOUT_PROPERTY_PRESSURE_PA,
        "burnout_property_T_sat_K": T_sat_K,
        "burnout_heated_length_m": BURNOUT_HEATED_LENGTH_M,
        "film_average_velocity_m_s": FILM_AVERAGE_VELOCITY_M_S,
        "film_thickness_m": FILM_THICKNESS_M,
        "liquid_absorptivity_1_m": LIQUID_ABSORPTIVITY_1_M,
        "radiation_transmitted_fraction": transmitted_fraction,
        "q_rad_transmitted_W_m2": q_rad_transmitted_W_m2,
        "q_rad_transmitted_MW_m2": q_rad_transmitted_W_m2 / 1.0e6,
        "fuel_property_model": "RP1 (RocketProps)",
        "fuel_liquid_density_kg_m3": rho_l_kg_m3,
        "fuel_vapor_density_kg_m3": rho_v_kg_m3,
        "fuel_latent_heat_J_kg": h_fg_J_kg,
        "fuel_surface_tension_N_m": surface_tension_N_m,
        "rocketprops_fallback_notes": rp1_fallback_summary(),
        "q_burnout_W_m2": q_burnout_W_m2,
        "q_burnout_MW_m2": q_burnout_W_m2 / 1.0e6,
        "burnout_margin_qbo_over_qtrans": q_burnout_W_m2 / q_rad_transmitted_W_m2
        if q_rad_transmitted_W_m2 > 0.0
        else math.inf,
        "burnout_warning": q_rad_transmitted_W_m2 > q_burnout_W_m2,
    }


def effective_length_m():
    diameter_in = DC_IN if LEFF_DIAMETER_SOURCE == "chamber" else DT_IN
    leff = 0.95 * diameter_in * 0.0254
    if APPLY_REFLECTIVE_WALL_CORRECTION:
        leff *= WALL_ABSORPTIVITY ** (-0.85)
    return leff


def finite_reference_chamber_leff_m():
    diameter_m = DC_IN * 0.0254
    length_m = LCHAMBER_TOTAL_TO_THROAT_IN * 0.0254
    volume_m3 = math.pi * diameter_m**2 * length_m / 4.0
    surrounding_area_m2 = math.pi * diameter_m * length_m + math.pi * diameter_m**2 / 2.0
    return 0.95 * (4.0 * volume_m3 / surrounding_area_m2)



def main():
    out_text, run_dir = run_cea(PC_PSI, OF_RATIO)
    cea_state = parse_cea_chamber_state(out_text)

    leff_m = effective_length_m()
    result = calculate_grissom_radiation(
        Tg_K=cea_state["Tg_K"],
        Twall_K=TWALL_K,
        pc_psi=PC_PSI,
        N_h2o=cea_state["N_H2O"],
        N_co2=cea_state["N_CO2"],
        leff_m=leff_m,
    )
    result["CEA_run_dir"] = str(run_dir)
    result["Leff_basis"] = f"0.95 * {LEFF_DIAMETER_SOURCE} diameter"
    result["reference_chamber_length_m"] = LCHAMBER_TOTAL_TO_THROAT_IN * 0.0254
    result["reference_chamber_0p95_4V_over_A_Leff_m"] = finite_reference_chamber_leff_m()
    result.update(calculate_liquid_film_burnout_check(result["q_rad_W_m2"]))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print("Grissom radiation calculation")
    print(f"CEA run directory: {run_dir}")
    print(f"O/F = {OF_RATIO:.3f}, Pc = {PC_PSI:.2f} psi ({result['Pc_atm']:.3f} atm)")
    print(f"Tg = {result['Tg_K']:.2f} K, Twall = {result['Twall_K']:.2f} K")
    print(f"N_H2O = {result['N_H2O']:.6g}, N_CO2 = {result['N_CO2']:.6g}")
    print(f"p_H2O = {result['p_H2O_atm']:.6g} atm, p_CO2 = {result['p_CO2_atm']:.6g} atm")
    print(f"Leff = {result['Leff_m']:.6f} m ({result['Leff_basis']})")
    print(
        "Reference chamber finite-cylinder Leff check = "
        f"{result['reference_chamber_0p95_4V_over_A_Leff_m']:.6f} m"
    )
    print(f"rho_H2O = {result['rho_H2O_atm_m']:.6g} atm-m")
    print(f"rho_CO2 = {result['rho_CO2_atm_m']:.6g} atm-m")
    print(f"epsilon_g = {result['epsilon_gas']:.6f}")
    print(f"q_rad = {result['q_rad_W_m2']:.3f} W/m2 = {result['q_rad_MW_m2']:.6f} MW/m2")
    print("\nLiquid film burnout warning check")
    print("---------------------------------")
    print(f"Correlation: {result['burnout_correlation']}")
    print(
        "q_rad transmitted = "
        f"{result['q_rad_transmitted_W_m2']:.3f} W/m2 = "
        f"{result['q_rad_transmitted_MW_m2']:.6f} MW/m2"
    )
    print(
        "q_burnout = "
        f"{result['q_burnout_W_m2']:.3f} W/m2 = "
        f"{result['q_burnout_MW_m2']:.6f} MW/m2"
    )
    print(f"burnout margin q_bo/q_trans = {result['burnout_margin_qbo_over_qtrans']:.3f}")
    print(f"burnout warning = {result['burnout_warning']}")
    print(f"fuel property model = {result['fuel_property_model']}")
    print(f"RocketProps fallbacks = {result['rocketprops_fallback_notes']}")
    print(f"Saved output CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
