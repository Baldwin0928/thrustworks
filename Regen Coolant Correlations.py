import math


def haaland_friction_factor(reynolds_number, hydraulic_diameter, roughness=0.0):
    """
    Darcy friction factor using the explicit Haaland approximation.
    For laminar flow, returns 64/Re.
    """
    if reynolds_number <= 0.0 or hydraulic_diameter <= 0.0:
        return math.nan

    if reynolds_number < 2300.0:
        return 64.0 / reynolds_number

    relative_roughness = roughness / hydraulic_diameter
    inv_sqrt_f = -1.8 * math.log10(
        (relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds_number
    )
    return 1.0 / (inv_sqrt_f**2)


def gnielinski_nusselt(reynolds_number, prandtl_number, friction_factor, allow_low_re=False):
    """
    Gnielinski Nusselt-number correlation for turbulent internal flow.
    Nu = [(f/8)(Re - 1000)Pr] /
         [1 + 12.7(f/8)^0.5(Pr^(2/3)-1)]
    """
    if (
        prandtl_number <= 0.0
        or friction_factor <= 0.0
        or (reynolds_number <= 3000.0 and not allow_low_re)
    ):
        return math.nan

    numerator = (friction_factor / 8.0) * (reynolds_number - 1000.0) * prandtl_number
    denominator = (
        1.0
        + 12.7
        * math.sqrt(friction_factor / 8.0)
        * (prandtl_number ** (2.0 / 3.0) - 1.0)
    )
    return numerator / denominator

def kerosene_nusselt(
    reynolds_number,
    prandtl_number,
    coolant_temperature,
    wall_temperature,
    allow_low_re=False,
):
    """
    Kerosene coolant-side Nusselt-number relation.
    Nu = 0.021 Re^0.8 Pr^0.4 (0.64 + 0.36 Tc/Twc)
    """
    if (
        prandtl_number <= 0.0
        or wall_temperature <= 0.0
        or (reynolds_number <= 3000.0 and not allow_low_re)
    ):
        return math.nan

    wall_factor = 0.64 + 0.36 * (coolant_temperature / wall_temperature)
    return 0.021 * (reynolds_number**0.8) * (prandtl_number**0.4) * wall_factor


def dittus_boelter_heating_nusselt(
    reynolds_number,
    prandtl_number,
    allow_low_re=False,
):
    """
    Dittus-Boelter heating Nusselt-number relation.

    Nu = 0.023 Re^0.8 Pr^0.4
    """
    if prandtl_number <= 0.0 or (reynolds_number <= 3000.0 and not allow_low_re):
        return math.nan

    return 0.023 * (reynolds_number**0.8) * (prandtl_number**0.4)


def sieder_tate_nusselt(
    reynolds_number,
    prandtl_number,
    viscosity_bulk,
    viscosity_wall,
    allow_low_re=False,
):
    """
    Sieder-Tate turbulent internal-flow Nusselt-number relation.
    Nu = 0.027 Re^0.8 Pr^(1/3) (mu_bulk / mu_wall)^0.14
    """
    if (
        prandtl_number <= 0.0
        or viscosity_bulk <= 0.0
        or viscosity_wall <= 0.0
        or (reynolds_number <= 10000.0 and not allow_low_re)
    ):
        return math.nan

    viscosity_ratio = viscosity_bulk / viscosity_wall
    return (
        0.027
        * (reynolds_number**0.8)
        * (prandtl_number ** (1.0 / 3.0))
        * (viscosity_ratio**0.14)
    )


if __name__ == "__main__":
    Re = 20000.0
    Pr = 4.0
    Dh = 1.5e-3
    roughness = 0.0
    T_c = 300.0
    T_wall = 600.0
    mu_bulk = 1.6e-3
    mu_wall = 0.7e-3

    f = haaland_friction_factor(Re, Dh, roughness)
    print(f"Haaland Darcy f:        {f:.6f}")
    print(f"Gnielinski Nu:          {gnielinski_nusselt(Re, Pr, f):.3f}")
    print(f"Kerosene Nu:            {kerosene_nusselt(Re, Pr, T_c, T_wall):.3f}")
    print(f"Dittus-Boelter Nu:      {dittus_boelter_heating_nusselt(Re, Pr):.3f}")
    print(f"Sieder-Tate Nu:         {sieder_tate_nusselt(Re, Pr, mu_bulk, mu_wall):.3f}")
