import os
import math
import csv
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Bartz Heat Flux Calculator
# Chamber pressure
Pc_psi = 300.0

# Characteristic velocity
cstar_m_s = 1779.870546

# CEA chamber / stagnation gas properties used by Bartz
Tc_K = 3241.5
gamma = 1.1692
Cp_J_kg_K = 4231.8
mu_Pa_s = 9.84e-5
Pr_gas = 0.3937

# Optional local CEA station-property anchors for gas state
USE_CEA_GAS_PROPERTIES = False
CEA_OF_RATIO = 2.0
CEA_RESULTS_CSV = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "CEA Run",
    "CEA_Results",
    "run_2026-05-15_19-13-49",
    "lox_rp1_cea_sizing_and_bartz_outputs.csv",
)

# Hot-gas-side chamber wall temperature
# This is NOT coolant temperature.
# This is the inner wall/gas-side wall temperature which
# will significantly affect your heat flux values and should be
# treated with caution.
Twg_K = 1200.0
Twg_sweep_K = np.arange(800.0, 1400.0 + 1.0, 100.0)

# Bartz options
USE_CHAMBER_FROZEN_BARTZ_PROPS = True
TAW_RECOVERY_FORMULATION = "stagnation"
TAW_RECOVERY_PR_SOURCE = "bartz_transport"
MODIFIED_BARTZ_TEMP_EXPONENT = 0.2
APPLY_MODIFIED_BARTZ_TEMP_CORRECTION = False
CHAMBER_FROZEN_BARTZ_PROPS_CSV = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "CEA Run",
    "CEA_Results",
    "bartz_frozen_props",
    "lox_rp1_chamber_frozen_bartz_props.csv",
)

# Geometry inputs of the engine
At_in2 = 5.927401
Ae_in2 = 22.720682

Dc_in = 6.143
Dt_in = math.sqrt(4.0 * At_in2 / math.pi)
De_in = math.sqrt(4.0 * Ae_in2 / math.pi)

Lc_in = 9.067759
Lconv_in = 1.697851
Ldiv_in = 4.910214

theta_conv_deg = 45.0
theta_div_deg = 15.0

# Throat radius of curvature, R, in inches
# Bartz needs a finite throat radius of curvature.
# Typical starting assumption: R_throat = 0.5 to 1.5 times throat radius
# Here I'm just using: R = throat radius
R_throat_in = Dt_in / 2.0

# Discretization
N_chamber = 80
N_converging = 60
N_diverging = 120

csv_output_name = "bartz_axial_heat_flux_outputs.csv"

# Constants and Unit Conversions
G0_FT_S2 = 32.174
IN_TO_M = 0.0254
IN2_TO_M2 = IN_TO_M ** 2
M_TO_FT = 3.280839895
K_TO_R = 9.0 / 5.0

# 1 Pa*s = 0.0559974 lbm/(in*s)
PA_S_TO_LBM_IN_S = 0.0559974

# 1 Btu/(lbm*R) = 4186.8 J/(kg*K)
J_KG_K_TO_BTU_LBM_R = 1.0 / 4186.8

# 1 Btu/(in^2*s*R) to W/(m^2*K)
BTU_IN2_S_R_TO_W_M2_K = 1055.05585262 / 0.00064516 * (9.0 / 5.0)

# 1 Btu/(in^2*s) to W/m^2
BTU_IN2_S_TO_W_M2 = 1055.05585262 / 0.00064516

# 1 W/m^2 to MW/m^2
W_M2_TO_MW_M2 = 1.0e-6

# Unit Conversions for Bartz Imperial Form
cstar_ft_s = cstar_m_s * M_TO_FT
Tc_R = Tc_K * K_TO_R
Twg_R = Twg_K * K_TO_R
Cp_Btu_lbm_R = Cp_J_kg_K * J_KG_K_TO_BTU_LBM_R
mu_lbm_in_s = mu_Pa_s * PA_S_TO_LBM_IN_S
recovery_factor = Pr_gas ** (1.0 / 3.0)


# Helper Functions
def safe_float(value, fallback=np.nan):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if np.isnan(parsed):
        return fallback
    return parsed


def find_latest_cea_results_csv():
    cea_root = os.path.join(os.path.dirname(SCRIPT_DIR), "CEA Run", "CEA_Results")
    if not os.path.isdir(cea_root):
        return None

    matches = []
    for root, _, files in os.walk(cea_root):
        for file_name in files:
            if file_name.endswith(".csv") and "cea" in file_name.lower():
                matches.append(os.path.join(root, file_name))

    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_chamber_frozen_bartz_props():
    fallback = {
        "Cp_J_kg_K": 2141.9,
        "mu_Pa_s": 9.843e-5,
        "Pr": 0.5944,
        "source": "hardcoded CEA frozen fallback",
    }
    if not os.path.isfile(CHAMBER_FROZEN_BARTZ_PROPS_CSV):
        return fallback

    try:
        with open(CHAMBER_FROZEN_BARTZ_PROPS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return fallback

    if len(rows) == 0:
        return fallback

    row = rows[0]
    return {
        "Cp_J_kg_K": safe_float(row.get("Cp_fr_chamber_J_kg_K"), fallback["Cp_J_kg_K"]),
        "mu_Pa_s": safe_float(row.get("mu_chamber_Pa_s"), fallback["mu_Pa_s"]),
        "Pr": safe_float(row.get("Pr_fr_chamber"), fallback["Pr"]),
        "source": str(row.get("source", "CEA chamber frozen transport CSV")),
    }


CHAMBER_FROZEN_BARTZ_PROPS = load_chamber_frozen_bartz_props()


def lerp(a, b, frac):
    return a + (b - a) * min(max(frac, 0.0), 1.0)


def load_cea_gas_property_anchors():
    fallback_exit_mach = solve_mach_from_area_ratio(Ae_in2 / At_in2, gamma, "supersonic")
    fallback_exit_static_K = static_gas_temperature_K(fallback_exit_mach, gamma, Tc_K)

    fallback = {
        "enabled": False,
        "source": "constant fallback",
        "path": "",
        "of_ratio": np.nan,
        "cstar_m_per_s": cstar_m_s,
        "Mach_exit": fallback_exit_mach,
        "chamber": {
            "T_static_K": Tc_K,
            "gamma": gamma,
            "Cp_J_kg_K": Cp_J_kg_K,
            "mu_Pa_s": mu_Pa_s,
            "Pr": Pr_gas,
        },
        "throat": {
            "T_static_K": Tc_K / (1.0 + ((gamma - 1.0) / 2.0)),
            "gamma": gamma,
            "Cp_J_kg_K": Cp_J_kg_K,
            "mu_Pa_s": mu_Pa_s,
            "Pr": Pr_gas,
        },
        "exit": {
            "T_static_K": fallback_exit_static_K,
            "gamma": gamma,
            "Cp_J_kg_K": Cp_J_kg_K,
            "mu_Pa_s": mu_Pa_s,
            "Pr": Pr_gas,
        },
    }

    if not USE_CEA_GAS_PROPERTIES:
        return fallback

    csv_path = CEA_RESULTS_CSV
    if not os.path.isfile(csv_path):
        csv_path = find_latest_cea_results_csv()
    if csv_path is None or not os.path.isfile(csv_path):
        return fallback

    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return fallback

    if len(rows) == 0 or "O/F" not in rows[0]:
        return fallback

    selected_row = min(
        rows,
        key=lambda row: abs(safe_float(row.get("O/F"), CEA_OF_RATIO) - CEA_OF_RATIO),
    )

    def anchor(prefix, default_t):
        return {
            "T_static_K": safe_float(selected_row.get(default_t), fallback["chamber"]["T_static_K"]),
            "gamma": safe_float(selected_row.get(f"gamma_{prefix}"), gamma),
            "Cp_J_kg_K": safe_float(selected_row.get(f"Cp_{prefix}_J_per_kg_K"), Cp_J_kg_K),
            "mu_Pa_s": safe_float(selected_row.get(f"mu_{prefix}_Pa_s"), mu_Pa_s),
            "Pr": safe_float(selected_row.get(f"Pr_{prefix}"), Pr_gas),
        }

    return {
        "enabled": True,
        "source": "CEA CSV anchors",
        "path": csv_path,
        "of_ratio": safe_float(selected_row.get("O/F"), CEA_OF_RATIO),
        "cstar_m_per_s": safe_float(selected_row.get("cstar_m_per_s"), cstar_m_s),
        "Mach_exit": safe_float(selected_row.get("Mach_exit"), np.nan),
        "chamber": anchor("chamber", "Tc_K"),
        "throat": anchor("throat", "Tt_K"),
        "exit": anchor("exit", "Te_K"),
    }


def interpolate_gas_anchor(a, b, frac, anchors, region, mapping_basis, mach_value):
    return {
        "T_static_K": lerp(a["T_static_K"], b["T_static_K"], frac),
        "gamma": lerp(a["gamma"], b["gamma"], frac),
        "Cp_J_kg_K": lerp(a["Cp_J_kg_K"], b["Cp_J_kg_K"], frac),
        "mu_Pa_s": lerp(a["mu_Pa_s"], b["mu_Pa_s"], frac),
        "Pr": lerp(a["Pr"], b["Pr"], frac),
        "Mach": mach_value,
        "cea_map_fraction": min(max(frac, 0.0), 1.0),
        "cstar_m_per_s": anchors["cstar_m_per_s"],
        "T_sigma_ref_K": anchors["chamber"]["T_static_K"],
        "gas_property_region": region,
        "gas_property_mapping": mapping_basis,
        "gas_property_source": anchors["source"],
        "cea_of_ratio": anchors["of_ratio"],
        "cea_csv_path": anchors["path"],
    }


def area_mach_relation(M, gamma_value):
    term1 = 1.0 / M
    term2 = (2.0 / (gamma_value + 1.0)) * (
        1.0 + ((gamma_value - 1.0) / 2.0) * M**2
    )
    exponent = (gamma_value + 1.0) / (2.0 * (gamma_value - 1.0))
    return term1 * term2**exponent


def solve_mach_from_area_ratio(area_ratio_value, gamma_value, branch):
    if abs(area_ratio_value - 1.0) < 1e-8:
        return 1.0

    if branch == "subsonic":
        low = 1e-6
        high = 0.999999
    elif branch == "supersonic":
        low = 1.000001
        high = 20.0
    else:
        raise ValueError("branch must be 'subsonic' or 'supersonic'")

    for _ in range(200):
        mid = 0.5 * (low + high)
        f_mid = area_mach_relation(mid, gamma_value) - area_ratio_value
        f_low = area_mach_relation(low, gamma_value) - area_ratio_value

        if f_low * f_mid <= 0:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)


def radius_at_x(x_in):
    # Local radius with x = 0 at injector face.
    rt_in = Dt_in / 2.0
    rc_in = Dc_in / 2.0
    re_in = De_in / 2.0

    x_throat_local_in = Lc_in + Lconv_in
    x_exit_local_in = Lc_in + Lconv_in + Ldiv_in

    if x_in <= Lc_in:
        return rc_in
    if x_in <= x_throat_local_in:
        s = (x_in - Lc_in) / Lconv_in
        return rc_in + s * (rt_in - rc_in)
    if x_in <= x_exit_local_in:
        s = (x_in - x_throat_local_in) / Ldiv_in
        return rt_in + s * (re_in - rt_in)
    return re_in


def area_at_x(x_in):
    r = radius_at_x(x_in)
    return math.pi * r**2


def mach_at_x(x_in):
    A_in2 = area_at_x(x_in)
    area_ratio = A_in2 / At_in2
    x_throat_local_in = Lc_in + Lconv_in

    if x_in < x_throat_local_in:
        return solve_mach_from_area_ratio(area_ratio, gamma, branch="subsonic")
    if abs(x_in - x_throat_local_in) < 1e-8:
        return 1.0
    return solve_mach_from_area_ratio(area_ratio, gamma, branch="supersonic")


def gas_props_at_station(x_in_value, x_throat_in, x_exit_in, anchors, area_ratio_override=None):
    if area_ratio_override is None:
        area_ratio = area_at_x(x_in_value) / At_in2
    else:
        area_ratio = max(float(area_ratio_override), 1.0)
    chamber_area_ratio = (math.pi * (Dc_in / 2.0) ** 2) / At_in2
    exit_area_ratio = Ae_in2 / At_in2

    if x_in_value < Lc_in:
        a = anchors["chamber"]
        b = anchors["chamber"]
        frac = 0.0
        region = "chamber_barrel"
        branch = "subsonic"
        mapping_basis = "constant_chamber_with_area_mach"
    elif x_in_value <= x_throat_in:
        a = anchors["chamber"]
        b = anchors["throat"]
        frac = (chamber_area_ratio - area_ratio) / max(chamber_area_ratio - 1.0, 1.0e-12)
        region = "converging_to_throat"
        branch = "subsonic"
        mapping_basis = "mach_fraction_chamber_to_throat"
    else:
        a = anchors["throat"]
        b = anchors["exit"]
        frac = (area_ratio - 1.0) / max(exit_area_ratio - 1.0, 1.0e-12)
        region = "diverging_to_exit"
        branch = "supersonic"
        mapping_basis = "mach_fraction_throat_to_exit"

    frac = min(max(frac, 0.0), 1.0)
    mach_value = 1.0

    for _ in range(8):
        gamma_value = lerp(a["gamma"], b["gamma"], frac)
        if abs(area_ratio - 1.0) < 1.0e-8:
            mach_value = 1.0
        else:
            mach_value = solve_mach_from_area_ratio(area_ratio, gamma_value, branch)

        if region == "chamber_barrel":
            new_frac = 0.0
        elif region == "converging_to_throat":
            chamber_mach = solve_mach_from_area_ratio(
                chamber_area_ratio,
                anchors["chamber"]["gamma"],
                "subsonic",
            )
            new_frac = (mach_value - chamber_mach) / max(1.0 - chamber_mach, 1.0e-12)
        else:
            exit_mach = anchors["Mach_exit"]
            if np.isnan(exit_mach):
                exit_mach = solve_mach_from_area_ratio(
                    exit_area_ratio,
                    anchors["exit"]["gamma"],
                    "supersonic",
                )
            new_frac = (mach_value - 1.0) / max(exit_mach - 1.0, 1.0e-12)

        new_frac = min(max(new_frac, 0.0), 1.0)
        if abs(new_frac - frac) < 1.0e-5:
            frac = new_frac
            break
        frac = 0.5 * (frac + new_frac)

    return interpolate_gas_anchor(a, b, frac, anchors, region, mapping_basis, mach_value)


def chamber_frozen_bartz_props(base_props):
    props = dict(base_props)
    props["Cp_J_kg_K"] = CHAMBER_FROZEN_BARTZ_PROPS["Cp_J_kg_K"]
    props["mu_Pa_s"] = CHAMBER_FROZEN_BARTZ_PROPS["mu_Pa_s"]
    props["Pr"] = CHAMBER_FROZEN_BARTZ_PROPS["Pr"]
    props["bartz_property_source"] = CHAMBER_FROZEN_BARTZ_PROPS["source"]
    return props


def taw_recovery_props(gas_state_props, bartz_transport_props=None):
    props = dict(gas_state_props)
    if TAW_RECOVERY_PR_SOURCE == "bartz_transport" and bartz_transport_props is not None:
        props["Pr"] = bartz_transport_props.get("Pr", props.get("Pr", Pr_gas))
    return props


def static_gas_temperature_K(M, gamma_value=gamma, t0_value_K=Tc_K):
    return t0_value_K / (1.0 + ((gamma_value - 1.0) / 2.0) * M**2)


def adiabatic_wall_temperature_K(M, gas_props=None):
    if gas_props is None:
        gas_props = {}
    gamma_value = gas_props.get("gamma", gamma)
    pr_value = max(gas_props.get("Pr", Pr_gas), 1.0e-12)
    t_static_value_K = gas_props.get("T_static_K")
    t0_value_K = gas_props.get("T_sigma_ref_K", Tc_K)

    a = (gamma_value - 1.0) / 2.0
    recovery = pr_value ** (1.0 / 3.0)
    if TAW_RECOVERY_FORMULATION == "stagnation":
        return t0_value_K * (1.0 + recovery * a * M**2) / (1.0 + a * M**2)

    if t_static_value_K is None:
        t_static_value_K = static_gas_temperature_K(M, gamma_value, t0_value_K)
    return t_static_value_K * (1.0 + recovery * a * M**2)


def sigma_correction(M, Twg_value_K=Twg_K, gamma_value=gamma, T_ref_value_K=Tc_K):
    a = (gamma_value - 1.0) / 2.0
    Twg_value_R = Twg_value_K * K_TO_R
    T_ref_R = T_ref_value_K * K_TO_R
    term1 = (0.5 * (Twg_value_R / T_ref_R) * (1.0 + a * M**2) + 0.5) ** 0.68
    term2 = (1.0 + a * M**2) ** 0.12
    return 1.0 / (term1 * term2)


def bartz_hg_W_m2_K(
    x_in_value,
    Twg_value_K=Twg_K,
    gas_props=None,
    apply_temp_correction=False,
    area_in2_override=None,
):
    if gas_props is None:
        gas_props = {}
    gamma_value = gas_props.get("gamma", gamma)
    cp_value = gas_props.get("Cp_J_kg_K", Cp_J_kg_K)
    mu_value = gas_props.get("mu_Pa_s", mu_Pa_s)
    pr_value = max(gas_props.get("Pr", Pr_gas), 1.0e-12)
    cstar_value = gas_props.get("cstar_m_per_s", cstar_m_s)
    t_ref_value_K = gas_props.get("T_sigma_ref_K", Tc_K)

    A_in2 = area_at_x(x_in_value) if area_in2_override is None else float(area_in2_override)
    M = gas_props.get("Mach", np.nan)
    if np.isnan(M):
        M = mach_at_x(x_in_value)
    sigma = sigma_correction(M, Twg_value_K, gamma_value, t_ref_value_K)
    temp_correction = 1.0
    if apply_temp_correction:
        taw_for_correction = adiabatic_wall_temperature_K(M, gas_props)
        t_ref_correction = 0.5 * (taw_for_correction + Twg_value_K)
        temp_correction = (
            max(t_ref_correction, 1.0e-12) / max(taw_for_correction, 1.0e-12)
        ) ** MODIFIED_BARTZ_TEMP_EXPONENT

    area_factor = (At_in2 / A_in2) ** 0.9
    radius_factor = (Dt_in / R_throat_in) ** 0.1
    cp_btu_lbm_R = cp_value * J_KG_K_TO_BTU_LBM_R
    mu_lbm_in_s_value = mu_value * PA_S_TO_LBM_IN_S
    cstar_ft_s_value = cstar_value * M_TO_FT

    hg_btu_in2_s_R = (
        (0.026 / (Dt_in ** 0.2))
        * ((mu_lbm_in_s_value ** 0.2) * cp_btu_lbm_R / (pr_value ** 0.6))
        * (((Pc_psi * G0_FT_S2) / cstar_ft_s_value) ** 0.8)
        * radius_factor
        * area_factor
        * sigma
        * temp_correction
    )

    return hg_btu_in2_s_R * BTU_IN2_S_R_TO_W_M2_K


def bartz_hg_btu_in2_s_R(x_in_value, gas_props=None, area_in2_override=None):
    hg_W_m2_K = bartz_hg_W_m2_K(
        x_in_value,
        Twg_K,
        gas_props,
        apply_temp_correction=APPLY_MODIFIED_BARTZ_TEMP_CORRECTION,
        area_in2_override=area_in2_override,
    )
    return hg_W_m2_K / BTU_IN2_S_R_TO_W_M2_K


# Building Axial Grid
x_chamber = np.linspace(0.0, Lc_in, N_chamber, endpoint=False)
x_converging = np.linspace(Lc_in, Lc_in + Lconv_in, N_converging, endpoint=False)
x_diverging = np.linspace(Lc_in + Lconv_in, Lc_in + Lconv_in + Ldiv_in, N_diverging)

x_in = np.concatenate([x_chamber, x_converging, x_diverging])
x_throat_in = Lc_in + Lconv_in
x_exit_in = Lc_in + Lconv_in + Ldiv_in
cea_anchors = load_cea_gas_property_anchors()


# Calculate axial profiles
radius_in = np.zeros_like(x_in)
area_in2 = np.zeros_like(x_in)
area_ratio = np.zeros_like(x_in)
mach = np.zeros_like(x_in)
sigma = np.zeros_like(x_in)
Taw_K = np.zeros_like(x_in)

hg_btu_in2_s_R = np.zeros_like(x_in)
hg_W_m2_K = np.zeros_like(x_in)
hg_kW_m2_K = np.zeros_like(x_in)

q_btu_in2_s = np.zeros_like(x_in)
q_W_m2 = np.zeros_like(x_in)
q_MW_m2 = np.zeros_like(x_in)

sweep_sigma = {float(Twg): np.zeros_like(x_in) for Twg in Twg_sweep_K}
sweep_hg_kW_m2_K = {float(Twg): np.zeros_like(x_in) for Twg in Twg_sweep_K}
sweep_q_MW_m2 = {float(Twg): np.zeros_like(x_in) for Twg in Twg_sweep_K}

raw_Cp_J_kg_K = np.zeros_like(x_in)
raw_mu_Pa_s = np.zeros_like(x_in)
raw_Pr = np.zeros_like(x_in)
bartz_Cp_J_kg_K = np.zeros_like(x_in)
bartz_mu_Pa_s = np.zeros_like(x_in)
bartz_Pr = np.zeros_like(x_in)
cea_map_fraction = np.zeros_like(x_in)
gas_property_region = []
gas_property_mapping = []

for i, x in enumerate(x_in):
    radius_in[i] = radius_at_x(x)
    area_in2[i] = area_at_x(x)
    area_ratio[i] = area_in2[i] / At_in2

    gas_props = gas_props_at_station(
        x,
        x_throat_in,
        x_exit_in,
        cea_anchors,
        area_ratio_override=area_ratio[i],
    )
    bartz_props = (
        chamber_frozen_bartz_props(gas_props)
        if USE_CHAMBER_FROZEN_BARTZ_PROPS
        else gas_props
    )
    taw_props = taw_recovery_props(gas_props, bartz_props)

    mach[i] = gas_props["Mach"]
    Taw_K[i] = adiabatic_wall_temperature_K(mach[i], taw_props)
    sigma[i] = sigma_correction(
        mach[i],
        Twg_K,
        gas_props["gamma"],
        gas_props["T_sigma_ref_K"],
    )

    hg_W_m2_K[i] = bartz_hg_W_m2_K(
        x,
        Twg_K,
        bartz_props,
        apply_temp_correction=APPLY_MODIFIED_BARTZ_TEMP_CORRECTION,
        area_in2_override=area_in2[i],
    )
    hg_btu_in2_s_R[i] = hg_W_m2_K[i] / BTU_IN2_S_R_TO_W_M2_K
    hg_kW_m2_K[i] = hg_W_m2_K[i] / 1000.0

    q_W_m2[i] = hg_W_m2_K[i] * (Taw_K[i] - Twg_K)
    q_btu_in2_s[i] = q_W_m2[i] / BTU_IN2_S_TO_W_M2
    q_MW_m2[i] = q_W_m2[i] * W_M2_TO_MW_M2

    for Twg_sweep in Twg_sweep_K:
        Twg_sweep = float(Twg_sweep)
        sweep_sigma[Twg_sweep][i] = sigma_correction(
            mach[i],
            Twg_sweep,
            gas_props["gamma"],
            gas_props["T_sigma_ref_K"],
        )
        sweep_hg_W_m2_K = bartz_hg_W_m2_K(
            x,
            Twg_sweep,
            bartz_props,
            apply_temp_correction=APPLY_MODIFIED_BARTZ_TEMP_CORRECTION,
            area_in2_override=area_in2[i],
        )
        sweep_hg_kW_m2_K[Twg_sweep][i] = sweep_hg_W_m2_K / 1000.0
        sweep_q_MW_m2[Twg_sweep][i] = sweep_hg_W_m2_K * (Taw_K[i] - Twg_sweep) * W_M2_TO_MW_M2

    raw_Cp_J_kg_K[i] = gas_props["Cp_J_kg_K"]
    raw_mu_Pa_s[i] = gas_props["mu_Pa_s"]
    raw_Pr[i] = gas_props["Pr"]
    bartz_Cp_J_kg_K[i] = bartz_props["Cp_J_kg_K"]
    bartz_mu_Pa_s[i] = bartz_props["mu_Pa_s"]
    bartz_Pr[i] = bartz_props["Pr"]
    cea_map_fraction[i] = gas_props["cea_map_fraction"]
    gas_property_region.append(gas_props["gas_property_region"])
    gas_property_mapping.append(gas_props["gas_property_mapping"])


# Saving to CSV
with open(csv_output_name, mode="w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "x_in",
        "x_from_throat_in",
        "radius_in",
        "diameter_in",
        "area_in2",
        "area_m2",
        "A_over_At",
        "Mach",
        "Taw_K",
        "Twg_K",
        "sigma",
        "hg_Btu_per_in2_s_R",
        "hg_W_per_m2_K",
        "hg_kW_per_m2_K",
        "q_Btu_per_in2_s",
        "q_W_per_m2",
        "q_MW_per_m2",
        "Cp_gas_J_per_kg_K",
        "mu_gas_Pa_s",
        "Pr_gas",
        "raw_Cp_gas_J_per_kg_K",
        "raw_mu_gas_Pa_s",
        "raw_Pr_gas",
        "bartz_property_basis",
        "bartz_property_source",
        "T_sigma_ref_K",
        "Taw_recovery_formulation",
        "Taw_recovery_Pr_source",
        "gas_property_region",
        "gas_property_mapping",
        "cea_map_fraction",
        "gas_property_source",
        "cea_of_ratio",
        "cea_csv_path",
    ])

    for i in range(len(x_in)):
        writer.writerow([
            x_in[i],
            x_in[i] - x_throat_in,
            radius_in[i],
            2.0 * radius_in[i],
            area_in2[i],
            area_in2[i] * IN2_TO_M2,
            area_ratio[i],
            mach[i],
            Taw_K[i],
            Twg_K,
            sigma[i],
            hg_btu_in2_s_R[i],
            hg_W_m2_K[i],
            hg_kW_m2_K[i],
            q_btu_in2_s[i],
            q_W_m2[i],
            q_MW_m2[i],
            bartz_Cp_J_kg_K[i],
            bartz_mu_Pa_s[i],
            bartz_Pr[i],
            raw_Cp_J_kg_K[i],
            raw_mu_Pa_s[i],
            raw_Pr[i],
            "chamber_frozen" if USE_CHAMBER_FROZEN_BARTZ_PROPS else "station_cea",
            CHAMBER_FROZEN_BARTZ_PROPS["source"] if USE_CHAMBER_FROZEN_BARTZ_PROPS else cea_anchors["source"],
            cea_anchors["chamber"]["T_static_K"],
            TAW_RECOVERY_FORMULATION,
            TAW_RECOVERY_PR_SOURCE,
            gas_property_region[i],
            gas_property_mapping[i],
            cea_map_fraction[i],
            cea_anchors["source"],
            cea_anchors["of_ratio"],
            cea_anchors["path"],
        ])

print(f"\nSaved Bartz axial outputs to: {csv_output_name}")


# Summary
idx_max_q = np.nanargmax(q_MW_m2)
idx_throat = np.nanargmin(np.abs(x_in - x_throat_in))

print("BARTZ HEAT FLUX SUMMARY")
print("\nInput Gas Properties")
print("------")
print(f"Pc                         : {Pc_psi:.3f} psi")
print(f"c*                         : {cstar_m_s:.6f} m/s")
print(f"Tc                         : {Tc_K:.3f} K")
print(f"gamma                      : {gamma:.6f}")
print(f"raw Cp                     : {Cp_J_kg_K:.3f} J/kg-K")
print(f"raw mu                     : {mu_Pa_s:.6e} Pa*s")
print(f"raw Pr                     : {Pr_gas:.6f}")
print(f"Bartz Cp                   : {bartz_Cp_J_kg_K[idx_throat]:.3f} J/kg-K")
print(f"Bartz mu                   : {bartz_mu_Pa_s[idx_throat]:.6e} Pa*s")
print(f"Bartz Pr                   : {bartz_Pr[idx_throat]:.6f}")
print(f"Bartz property basis       : {'chamber_frozen' if USE_CHAMBER_FROZEN_BARTZ_PROPS else 'station_cea'}")
print(f"Bartz property source      : {CHAMBER_FROZEN_BARTZ_PROPS['source'] if USE_CHAMBER_FROZEN_BARTZ_PROPS else cea_anchors['source']}")
print(f"gas-side wall temp, Twg    : {Twg_K:.3f} K")

print("\nGeometry")
print("------")
print(f"Chamber diameter, Dc       : {Dc_in:.6f} in")
print(f"Throat diameter, Dt        : {Dt_in:.6f} in")
print(f"Exit diameter, De          : {De_in:.6f} in")
print(f"Throat radius curvature, R : {R_throat_in:.6f} in")
print(f"Chamber length, Lc         : {Lc_in:.6f} in")
print(f"Converging length          : {Lconv_in:.6f} in")
print(f"Diverging length           : {Ldiv_in:.6f} in")
print(f"Injector face to throat    : {x_throat_in:.6f} in")
print(f"Injector face to exit      : {x_exit_in:.6f} in")

print("\nAt Throat")
print("------")
print(f"x                          : {x_in[idx_throat]:.6f} in")
print(f"x from throat              : {x_in[idx_throat] - x_throat_in:.6f} in")
print(f"Mach                       : {mach[idx_throat]:.6f}")
print(f"Taw                        : {Taw_K[idx_throat]:.3f} K")
print(f"sigma                      : {sigma[idx_throat]:.6f}")
print(f"hg                         : {hg_kW_m2_K[idx_throat]:.3f} kW/m^2-K")
print(f"q''                        : {q_MW_m2[idx_throat]:.6f} MW/m^2")

print("\nMaximum Heat Flux")
print("------")
print(f"x                          : {x_in[idx_max_q]:.6f} in")
print(f"x from throat              : {x_in[idx_max_q] - x_throat_in:.6f} in")
print(f"Mach                       : {mach[idx_max_q]:.6f}")
print(f"Taw                        : {Taw_K[idx_max_q]:.3f} K")
print(f"sigma                      : {sigma[idx_max_q]:.6f}")
print(f"hg                         : {hg_kW_m2_K[idx_max_q]:.3f} kW/m^2-K")
print(f"q''                        : {q_MW_m2[idx_max_q]:.6f} MW/m^2")


# Plots
fig, axs = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
fig.suptitle(
    "Bartz Heat Transfer Results Along Thrust Chamber",
    fontsize=16
)

x_plot = x_in - x_throat_in
sweep_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(Twg_sweep_K)))

# 1. Heat flux
ax = axs[0, 0]
for Twg_sweep, color in zip(Twg_sweep_K, sweep_colors):
    Twg_sweep = float(Twg_sweep)
    linewidth = 2.5 if abs(Twg_sweep - Twg_K) < 1.0e-9 else 1.6
    ax.plot(
        x_plot,
        sweep_q_MW_m2[Twg_sweep],
        linewidth=linewidth,
        color=color,
        label=f"Twg = {Twg_sweep:.0f} K",
    )
ax.axvline(0.0, linestyle="--", linewidth=1, color="0.35", label="Throat")
ax.set_title("Heat Flux vs Axial Location")
ax.set_xlabel("Axial Position from Throat (in)")
ax.set_ylabel("q'' (MW/m^2)")
ax.grid(True)
ax.legend(fontsize=8, ncol=2)

# 2. Gas-side heat transfer coefficient
ax = axs[0, 1]
for Twg_sweep, color in zip(Twg_sweep_K, sweep_colors):
    Twg_sweep = float(Twg_sweep)
    linewidth = 2.5 if abs(Twg_sweep - Twg_K) < 1.0e-9 else 1.6
    ax.plot(
        x_plot,
        sweep_hg_kW_m2_K[Twg_sweep],
        linewidth=linewidth,
        color=color,
        label=f"Twg = {Twg_sweep:.0f} K",
    )
ax.axvline(0.0, linestyle="--", linewidth=1, color="0.35", label="Throat")
ax.set_title("Gas-side Heat Transfer Coefficient")
ax.set_xlabel("Axial Position from Throat (in)")
ax.set_ylabel("h_g (kW/m^2-K)")
ax.grid(True)
ax.legend(fontsize=8, ncol=2)

# 3. Adiabatic wall temperature
ax = axs[1, 0]
ax.plot(x_plot, Taw_K, linewidth=2.5, color="black", label="Taw")
for Twg_sweep, color in zip(Twg_sweep_K, sweep_colors):
    Twg_sweep = float(Twg_sweep)
    linewidth = 1.5 if abs(Twg_sweep - Twg_K) < 1.0e-9 else 1.0
    ax.axhline(
        Twg_sweep,
        linestyle="--",
        linewidth=linewidth,
        color=color,
        label=f"Twg = {Twg_sweep:.0f} K",
    )
ax.axvline(0.0, linestyle="--", linewidth=1, color="0.35", label="Throat")
ax.set_title("Adiabatic Wall Temperature")
ax.set_xlabel("Axial Position from Throat (in)")
ax.set_ylabel("Temperature (K)")
ax.grid(True)
ax.legend(fontsize=8, ncol=2)

# 4. Mach number
ax = axs[1, 1]
ax.plot(x_plot, mach, linewidth=2)
ax.axvline(0.0, linestyle="--", linewidth=1, label="Throat")
ax.set_title("Estimated Mach Number")
ax.set_xlabel("Axial Position from Throat (in)")
ax.set_ylabel("Mach Number")
ax.grid(True)
ax.legend()

# 5. Internal contour
ax = axs[2, 0]
ax.plot(x_plot, 2.0 * radius_in, linewidth=2)
ax.axvline(0.0, linestyle="--", linewidth=1, label="Throat")
ax.set_title("Thrust Chamber Internal Diameter")
ax.set_xlabel("Axial Position from Throat (in)")
ax.set_ylabel("Local Diameter (in)")
ax.grid(True)
ax.legend()

# 6. Sigma correction
ax = axs[2, 1]
for Twg_sweep, color in zip(Twg_sweep_K, sweep_colors):
    Twg_sweep = float(Twg_sweep)
    linewidth = 2.5 if abs(Twg_sweep - Twg_K) < 1.0e-9 else 1.6
    ax.plot(
        x_plot,
        sweep_sigma[Twg_sweep],
        linewidth=linewidth,
        color=color,
        label=f"Twg = {Twg_sweep:.0f} K",
    )
ax.axvline(0.0, linestyle="--", linewidth=1, color="0.35", label="Throat")
ax.set_title("Bartz Sigma Correction")
ax.set_xlabel("Axial Position from Throat (in)")
ax.set_ylabel("Sigma")
ax.grid(True)
ax.legend(fontsize=8, ncol=2)

plt.show()