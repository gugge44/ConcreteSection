import math

try:
    import pint
    _PINT_AVAILABLE = True
except ImportError:
    _PINT_AVAILABLE = False

def _is_loadvector(value) -> bool:
    """Duck-type check for LoadVector. Works regardless of import path."""
    return hasattr(value, '_components') and hasattr(value, '_units') and hasattr(value, 'components')


def eng_format(value, sig_figs: int = 3, unit_fmt: str = "~P") -> str:
    """
    Convert a number to engineering notation with special rules:
    1. Small numbers (<1): Shift to higher exponent (e.g. 0.5e-3) if it results 
       in a clean decimal (<= 2 places).
    2. Thousands (1k-10k): Display as direct number with comma (e.g. 1,230) 
       instead of 1.23e3. Double digit thousands (10k+) remain as engineering (e.g. 12.3e3).
    
    Accepts plain numbers (int, float, str), pint Quantities, or LoadVectors.

    For pint Quantities, the unit is appended using the given format string
    (default "~P" for short pretty: N/m^2 -> N/m²).
    
    For LoadVectors, each component is formatted individually and combined
    as a load decomposition string (e.g. "420 + 300 + 72 [kN]" or with
    basis labels "420·DL + 300·A + 72·S2 [kN/m²]").
    
    Common pint format strings:
        "~P"  short pretty   kN/m²
        "P"   long pretty    kilonewton/meter²
        "~L"  short LaTeX    \\frac{kN}{m^{2}}
        "~H"  short HTML     kN/m<sup>2</sup>
        "~D"  short default  kN/m**2
    """
    # --- LoadVector handling (duck-typed, import-path independent) ---
    if _is_loadvector(value):
        return _eng_format_loadvector(value, sig_figs, unit_fmt)

    # --- Extract magnitude and units from pint Quantity ---
    units = None
    if _PINT_AVAILABLE and isinstance(value, pint.Quantity):
        units = value.units
        value = value.magnitude

    return _eng_format_scalar(value, sig_figs, units, unit_fmt)


def _eng_format_scalar(value, sig_figs: int, units=None, unit_fmt: str = "~P") -> str:
    """Format a scalar value (int, float, str) with optional pint units."""
    if isinstance(value, str):
        try:
            value = float(value)
        except:
            return value
    
    if value == 0:
        num_str = '0'
    else:
        sign = ''
        if value < 0:
            sign = '-'
            value = abs(value)
        
        exp10 = math.floor(math.log10(value))
        eng_exp = (exp10 // 3) * 3
        mantissa = value / (10 ** eng_exp)
        
        # Initial precision calculation
        mantissa_exp = math.floor(math.log10(mantissa))
        decimal_places = sig_figs - mantissa_exp - 1
        decimal_places = max(0, decimal_places)
        mantissa = round(mantissa, decimal_places)
        
        # Handle rounding up to 1000
        if mantissa >= 1000:
            mantissa = mantissa / 1000
            eng_exp += 3
            mantissa_exp = math.floor(math.log10(mantissa))
            decimal_places = sig_figs - mantissa_exp - 1
            decimal_places = max(0, decimal_places)
            mantissa = round(mantissa, decimal_places)
        
        # Handle rounding up to next order
        if mantissa >= 10 ** (mantissa_exp + 1):
            mantissa_exp += 1
            decimal_places = sig_figs - mantissa_exp - 1
            decimal_places = max(0, decimal_places)
            mantissa = round(mantissa, decimal_places)
        
        # --- RULE 1: CLEAN SMALL NUMBERS ---
        if value < 1:
            shifted_mantissa = mantissa / 1000
            shifted_exp = eng_exp + 3
            
            s_check = f"{shifted_mantissa:.10g}"
            if "." in s_check:
                actual_decimals = len(s_check.split(".")[1])
            else:
                actual_decimals = 0
                
            if actual_decimals <= 2:
                mantissa = shifted_mantissa
                eng_exp = shifted_exp
                decimal_places = actual_decimals

        # --- RULE 2: SINGLE DIGIT THOUSANDS (e.g. 1,230 instead of 1.23e3) ---
        if eng_exp == 3 and mantissa < 10:
            num_str = f"{sign}{int(mantissa * 1000):,}"
        else:
            # Format mantissa - strip trailing zeros after decimal point
            if decimal_places > 0:
                mantissa_str = f"{mantissa:.{decimal_places}f}"
                mantissa_str = mantissa_str.rstrip('0').rstrip('.')
            else:
                mantissa_str = f"{int(mantissa)}"
            
            if eng_exp == 0:
                num_str = f"{sign}{mantissa_str}"
            else:
                num_str = f"{sign}{mantissa_str}e{eng_exp}"

    # Append unit string if pint Quantity was provided
    if units is not None:
        unit_str = format(units, unit_fmt)
        if unit_str and unit_str != "dimensionless":
            return f"{num_str} {unit_str}"
    
    return num_str


def _eng_format_loadvector(lv, sig_figs: int = 3, unit_fmt: str = "~P") -> str:
    """Format a LoadVector as a decomposed expression.
    
    Single component:   420 kN [DL]
    Multiple:           420·DL + 300·A + 72·S2 [kN/m²]
    
    Each component magnitude is formatted through eng_format_scalar,
    then the shared pint unit is appended once at the end.
    """
    components = lv.components   # dict[str, float]
    units = lv.units

    # Unit string (shared across all components)
    unit_str = format(units, unit_fmt)
    if unit_str == "dimensionless":
        unit_str = ""

    if not components:
        return f"0 {unit_str}".strip()

    # Single component: cleaner format
    if len(components) == 1:
        key, mag = next(iter(components.items()))
        mag_str = _eng_format_scalar(mag, sig_figs)
        if unit_str:
            return f"{mag_str} {unit_str} [{key}]"
        return f"{mag_str} [{key}]"

    # Multiple components: vector notation
    terms = []
    for i, (key, mag) in enumerate(components.items()):
        mag_str = _eng_format_scalar(abs(mag), sig_figs)
        if i == 0:
            prefix = "-" if mag < 0 else ""
            terms.append(f"{prefix}{mag_str}·{key}")
        else:
            op = "- " if mag < 0 else "+ "
            terms.append(f"{op}{mag_str}·{key}")

    expr = ' '.join(terms)

    if unit_str:
        return f"{expr} [{unit_str}]"
    return expr


if __name__ == "__main__":
    def run_test(val, expected, description):
        result = eng_format(val, sig_figs=3)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status:<4} | Input: {str(val):<20} | Expected: {expected:<16} | Got: {result:<16} | {description}")

    print("=" * 100)
    print("PLAIN NUMBER TESTS")
    print("=" * 100)

    print("\n--- 1. Special Case: Single Digit Thousands (1k - 9.99k) ---")
    run_test(1234,   "1,230",   "1.234k -> Rounds to 3sf -> 1,230")
    run_test(5000,   "5,000",   "5k -> 5,000")
    run_test(-1234,  "-1,230",  "Negative 1.23k -> -1,230")
    
    print("\n--- 2. Transition to Double Digit Thousands (10k+) ---")
    run_test(9999,   "10e3",    "9.999k -> Rounds to 10e3") 
    run_test(12345,  "12.3e3",  "12k -> Double digit -> 12.3e3")
    
    print("\n--- 3. Clean Small Numbers (<1) ---")
    run_test(0.0005,  "0.5e-3", "500e-6 -> 0.5e-3")
    run_test(0.00053, "0.53e-3","530e-6 -> 0.53e-3")
    run_test(0.000532,"532e-6", "0.532 (3 dec) -> 532e-6")

    if _PINT_AVAILABLE:
        ureg = pint.UnitRegistry()

        print("\n" + "=" * 100)
        print("PINT QUANTITY TESTS")
        print("=" * 100)

        print("\n--- 4. Pint Quantities (default ~P format) ---")
        run_test(ureg.Quantity(1234, 'N/m**2'),    "1,230 N/m²",    "1234 N/m² -> 1,230 N/m²")
        run_test(ureg.Quantity(0.0005, 'kN'),       "0.5e-3 kN",    "0.5e-3 kN")
        run_test(ureg.Quantity(12345, 'mm'),         "12.3e3 mm",    "12.3e3 mm")
        run_test(ureg.Quantity(25.7, 'kN*m'),        "25.7 kN·m",    "25.7 kN·m")
        run_test(ureg.Quantity(-3.45e6, 'Pa'),       "-3.45e6 Pa",   "-3.45 MPa equiv")
        run_test(ureg.Quantity(0, 'kN'),             "0 kN",         "Zero with units")

        print("\n--- 5. Pint with custom format string ---")
        result_latex = eng_format(ureg.Quantity(1234, 'N/m**2'), sig_figs=3, unit_fmt="~L")
        print(f"     | LaTeX format: {result_latex}")
        result_long = eng_format(ureg.Quantity(1234, 'N/m**2'), sig_figs=3, unit_fmt="P")
        print(f"     | Long format:  {result_long}")

        print("\n--- 6. Dimensionless pint Quantity ---")
        run_test(ureg.Quantity(42.0, ''),            "42",            "Dimensionless -> no unit suffix")
    else:
        print("\npint not installed, skipping Quantity tests")

    try:
        from LoadPint import LoadRegistry

        reg = LoadRegistry()
        DL, SDL, A, S2, W, kN_m2, kN, m, m2 = reg.get('DL, SDL, A, S2, W, kN_m2, kN, m, m2')

        print("\n" + "=" * 100)
        print("LOADVECTOR TESTS")
        print("=" * 100)

        print("\n--- 7. Single-component LoadVector ---")
        gk = 3.5 * DL * kN_m2
        result = eng_format(gk)
        print(f"SHOW | {gk!r:<45} | eng_format -> {result}")

        gk_big = 1234 * DL * kN
        result = eng_format(gk_big)
        print(f"SHOW | {gk_big!r:<45} | eng_format -> {result}")

        print("\n--- 8. Multi-component LoadVector ---")
        w = 3.5 * DL * kN_m2 + 1.5 * SDL * kN_m2 + 2.5 * A * kN_m2
        result = eng_format(w)
        print(f"SHOW | {w!r:<45} | eng_format -> {result}")

        F = w * 120 * m2
        result = eng_format(F)
        print(f"SHOW | {F!r:<45} | eng_format -> {result}")

        print("\n--- 9. LoadVector with negative components ---")
        uplift = 2.0 * DL * kN_m2 + (-3.0) * W * kN_m2
        result = eng_format(uplift)
        print(f"SHOW | {uplift!r:<45} | eng_format -> {result}")

        print("\n--- 10. LoadVector with small/large magnitudes ---")
        small = 0.0005 * DL * kN_m2 + 0.00053 * A * kN_m2
        result = eng_format(small)
        print(f"SHOW | {small!r:<45} | eng_format -> {result}")

        big = 12345 * DL * kN + 5000 * A * kN + 72 * S2 * kN
        result = eng_format(big)
        print(f"SHOW | {big!r:<45} | eng_format -> {result}")

        print("\n--- 11. LoadVector dimensionless (basis only, no unit) ---")
        basis_only = 3.5 * DL + 2.5 * A
        result = eng_format(basis_only)
        print(f"SHOW | {basis_only!r:<45} | eng_format -> {result}")

        print("\n--- 12. Zero LoadVector ---")
        zero = 0 * DL * kN
        result = eng_format(zero)
        print(f"SHOW | zero LoadVector                                | eng_format -> {result}")

        print("\n--- 13. LaTeX format on LoadVector ---")
        result_latex = eng_format(F, sig_figs=3, unit_fmt="~L")
        print(f"SHOW | LaTeX: {result_latex}")

        print("\n--- 14. Combined result (pint Quantity from .uls_6_10) ---")
        uls = F.uls_6_10()
        result = eng_format(uls)
        print(f"SHOW | F.uls_6_10() = {uls:~.4g} | eng_format -> {result}")

    except ImportError:
        print("\nLoadPint not available, skipping LoadVector tests")