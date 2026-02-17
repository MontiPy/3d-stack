"""Process capability metrics and statistical utilities.

Provides Cp, Cpk, Pp, Ppk, DPMO, PPM, yield estimation, and percent
contribution analysis matching industrial standards (VisVSA / 3DCS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ProcessCapability:
    """Process capability metrics for a measurement.

    Attributes:
        usl: Upper specification limit.
        lsl: Lower specification limit.
        target: Target value (defaults to midpoint of limits).
        cp: Process capability index (spread only).
        cpk: Process capability index (with centering).
        pp: Process performance index (spread only, from actual data).
        ppk: Process performance index (with centering, from actual data).
        cpm: Taguchi capability index (accounts for target offset).
        ppm_upper: Parts per million exceeding USL.
        ppm_lower: Parts per million below LSL.
        ppm_total: Total PPM out of spec.
        dpmo: Defects per million opportunities.
        yield_percent: Estimated yield percentage.
        sigma_level: Estimated sigma level (Z-score).
        percent_out_of_spec: Percentage of samples outside spec.
        n_samples: Number of samples used.
        mean: Sample mean.
        std: Sample standard deviation.
    """
    usl: float
    lsl: float
    target: Optional[float] = None
    cp: float = 0.0
    cpk: float = 0.0
    pp: float = 0.0
    ppk: float = 0.0
    cpm: float = 0.0
    ppm_upper: float = 0.0
    ppm_lower: float = 0.0
    ppm_total: float = 0.0
    dpmo: float = 0.0
    yield_percent: float = 0.0
    sigma_level: float = 0.0
    percent_out_of_spec: float = 0.0
    n_samples: int = 0
    mean: float = 0.0
    std: float = 0.0

    def summary(self) -> str:
        lines = [
            "=== Process Capability ===",
            f"  Specification:  LSL={self.lsl:.6f}  USL={self.usl:.6f}",
            f"  Target:         {self.target:.6f}" if self.target is not None else "",
            f"  Sample mean:    {self.mean:.6f}",
            f"  Sample std:     {self.std:.6f}",
            f"  Cp:             {self.cp:.4f}",
            f"  Cpk:            {self.cpk:.4f}",
            f"  Pp:             {self.pp:.4f}",
            f"  Ppk:            {self.ppk:.4f}",
            f"  Cpm:            {self.cpm:.4f}",
            f"  Sigma level:    {self.sigma_level:.2f}",
            f"  Yield:          {self.yield_percent:.4f}%",
            f"  PPM total:      {self.ppm_total:.1f}",
            f"  DPMO:           {self.dpmo:.1f}",
            f"  Out of spec:    {self.percent_out_of_spec:.4f}%",
            f"  N samples:      {self.n_samples}",
        ]
        return "\n".join(line for line in lines if line)


def compute_process_capability(
    samples: np.ndarray,
    usl: float,
    lsl: float,
    target: Optional[float] = None,
    subgroup_size: int = 1,
) -> ProcessCapability:
    """Compute process capability metrics from sample data.

    Args:
        samples: 1D array of measurement values.
        usl: Upper specification limit.
        lsl: Lower specification limit.
        target: Target value. Defaults to midpoint of USL/LSL.
        subgroup_size: Subgroup size for within-group std estimation.
            If 1, Cp/Cpk use overall std (same as Pp/Ppk).

    Returns:
        ProcessCapability with all metrics computed.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    n = len(samples)
    if n < 2:
        raise ValueError("Need at least 2 samples for capability analysis")

    if target is None:
        target = (usl + lsl) / 2.0

    mean = float(np.mean(samples))
    std_overall = float(np.std(samples, ddof=1))

    # Within-subgroup std estimate (for Cp/Cpk)
    if subgroup_size > 1 and n >= subgroup_size * 2:
        n_groups = n // subgroup_size
        groups = samples[:n_groups * subgroup_size].reshape(n_groups, subgroup_size)
        ranges = np.ptp(groups, axis=1)
        # d2 constants for subgroup sizes (approximation)
        d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
                     7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        d2 = d2_table.get(subgroup_size, np.sqrt(2 * np.log(subgroup_size)) * 0.98)
        std_within = float(np.mean(ranges) / d2)
    else:
        std_within = std_overall

    spec_range = usl - lsl

    # Guard against zero std
    if std_within < 1e-15:
        std_within = 1e-15
    if std_overall < 1e-15:
        std_overall = 1e-15

    # Cp / Cpk (using within-group sigma)
    cp = spec_range / (6.0 * std_within)
    cpu = (usl - mean) / (3.0 * std_within)
    cpl = (mean - lsl) / (3.0 * std_within)
    cpk = min(cpu, cpl)

    # Pp / Ppk (using overall sigma)
    pp = spec_range / (6.0 * std_overall)
    ppu = (usl - mean) / (3.0 * std_overall)
    ppl = (mean - lsl) / (3.0 * std_overall)
    ppk = min(ppu, ppl)

    # Cpm (Taguchi) — accounts for deviation from target
    std_target = np.sqrt(std_overall ** 2 + (mean - target) ** 2)
    cpm = spec_range / (6.0 * std_target)

    # PPM and yield from actual data
    n_above = int(np.sum(samples > usl))
    n_below = int(np.sum(samples < lsl))
    n_out = n_above + n_below

    ppm_upper = (n_above / n) * 1_000_000 if n > 0 else 0.0
    ppm_lower = (n_below / n) * 1_000_000 if n > 0 else 0.0
    ppm_total = ppm_upper + ppm_lower
    dpmo = ppm_total  # For single CTQ, DPMO = PPM total

    percent_out = (n_out / n) * 100.0 if n > 0 else 0.0
    yield_pct = 100.0 - percent_out

    # Sigma level estimate from Cpk
    sigma_level = 3.0 * cpk if cpk > 0 else 0.0

    return ProcessCapability(
        usl=usl,
        lsl=lsl,
        target=target,
        cp=cp,
        cpk=cpk,
        pp=pp,
        ppk=ppk,
        cpm=cpm,
        ppm_upper=ppm_upper,
        ppm_lower=ppm_lower,
        ppm_total=ppm_total,
        dpmo=dpmo,
        yield_percent=yield_pct,
        sigma_level=sigma_level,
        percent_out_of_spec=percent_out,
        n_samples=n,
        mean=mean,
        std=std_overall,
    )


def percent_contribution(
    sensitivity: list[tuple[str, float]],
    tolerances: list[float],
) -> list[tuple[str, float]]:
    """Compute percent contribution of each tolerance to total variation.

    Uses RSS-style variance weighting: contribution_i = (sens_i * tol_i)^2 / total_var.

    Args:
        sensitivity: List of (name, sensitivity_value) tuples.
        tolerances: List of half-tolerance values (same order as sensitivity).

    Returns:
        List of (name, percent_contribution) sorted by contribution descending.
    """
    if len(sensitivity) != len(tolerances):
        raise ValueError("Sensitivity and tolerance lists must have same length")

    variances = []
    for (name, sens), tol in zip(sensitivity, tolerances):
        variances.append((name, (sens * tol) ** 2))

    total_var = sum(v for _, v in variances)
    if total_var < 1e-30:
        return [(name, 0.0) for name, _ in variances]

    pcts = [(name, (v / total_var) * 100.0) for name, v in variances]
    return sorted(pcts, key=lambda x: x[1], reverse=True)


def geo_factor(
    sensitivity: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Compute geometric amplification factor for each tolerance.

    GeoFactor = |sensitivity|. Values > 1 mean tolerance is amplified by
    geometry (lever arm effect). Values < 1 mean geometry mitigates the
    tolerance contribution.

    Args:
        sensitivity: List of (name, sensitivity_value) tuples.

    Returns:
        List of (name, geo_factor) sorted by factor descending.
    """
    factors = [(name, abs(sens)) for name, sens in sensitivity]
    return sorted(factors, key=lambda x: x[1], reverse=True)


def sample_distribution(
    rng: np.random.Generator,
    distribution,  # Distribution enum
    center: float,
    half_tol: float,
    sigma: float,
    n_samples: int,
    empirical_data: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate samples from any supported distribution.

    This is the central sampling function used by all MC analysis engines.

    Args:
        rng: NumPy random generator.
        distribution: Distribution enum value.
        center: Center value (nominal + midpoint shift).
        half_tol: Half of the total tolerance band.
        sigma: Sigma level for normal distribution.
        n_samples: Number of samples to generate.
        empirical_data: Optional array of empirical measurement data.

    Returns:
        1D array of samples.
    """
    from tolerance_stack.models import Distribution

    if distribution == Distribution.NORMAL:
        std = half_tol / sigma
        return rng.normal(loc=center, scale=std, size=n_samples)

    elif distribution == Distribution.UNIFORM:
        return rng.uniform(low=center - half_tol, high=center + half_tol, size=n_samples)

    elif distribution == Distribution.TRIANGULAR:
        return rng.triangular(
            left=center - half_tol, mode=center,
            right=center + half_tol, size=n_samples,
        )

    elif distribution == Distribution.WEIBULL_RIGHT:
        # Right-skewed Weibull with shape=2.5
        shape = 2.5
        scale = half_tol / 1.2  # Approximate to fit within tolerance band
        raw = rng.weibull(shape, size=n_samples) * scale
        raw = raw - np.mean(raw) + center
        return np.clip(raw, center - half_tol * 2, center + half_tol * 2)

    elif distribution == Distribution.WEIBULL_LEFT:
        # Left-skewed = mirrored right-skewed
        shape = 2.5
        scale = half_tol / 1.2
        raw = rng.weibull(shape, size=n_samples) * scale
        raw = -(raw - np.mean(raw)) + center
        return np.clip(raw, center - half_tol * 2, center + half_tol * 2)

    elif distribution == Distribution.LOGNORMAL:
        # Log-normal centered at 'center' with spread ~half_tol
        underlying_sigma = 0.3  # moderate skew
        raw = rng.lognormal(mean=0.0, sigma=underlying_sigma, size=n_samples)
        # Normalize to desired center and spread
        raw_mean = np.exp(underlying_sigma ** 2 / 2)
        raw_std = raw_mean * np.sqrt(np.exp(underlying_sigma ** 2) - 1)
        std = half_tol / sigma
        raw = (raw - raw_mean) / raw_std * std + center
        return raw

    elif distribution == Distribution.RAYLEIGH:
        # Rayleigh for radial tolerances, always positive
        scale = half_tol / 1.5  # ~mode near half_tol
        raw = rng.rayleigh(scale=scale, size=n_samples)
        # Shift so mean aligns with center
        raw = raw - np.mean(raw) + center
        return raw

    elif distribution == Distribution.BIMODAL:
        # Two equal peaks at ±half_tol from center
        std = half_tol * 0.15  # Narrow peaks
        half = n_samples // 2
        low_peak = rng.normal(loc=center - half_tol * 0.7, scale=std, size=half)
        high_peak = rng.normal(loc=center + half_tol * 0.7, scale=std, size=n_samples - half)
        samples = np.concatenate([low_peak, high_peak])
        rng.shuffle(samples)
        return samples

    elif distribution == Distribution.EMPIRICAL:
        if empirical_data is None or len(empirical_data) == 0:
            raise ValueError("Empirical distribution requires empirical_data")
        # Bootstrap resampling from provided data
        idx = rng.integers(0, len(empirical_data), size=n_samples)
        return np.asarray(empirical_data, dtype=float)[idx]

    else:
        raise ValueError(f"Unknown distribution: {distribution}")
