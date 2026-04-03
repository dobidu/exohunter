# Scientific Theory Behind ExoHunter

This document explains the astrophysics and signal processing concepts
that underpin the ExoHunter pipeline. It is written for Computer Science
students who may not have a background in astronomy.

---

## 1. What is an exoplanet?

An **exoplanet** is a planet that orbits a star other than our Sun. The
first confirmed exoplanet around a Sun-like star was discovered in 1995,
and as of 2025 we know of more than 5,700. Most were found not by
*seeing* the planet directly (they are far too faint), but by detecting
the subtle effect that the planet has on its host star.

---

## 2. The transit method

### 2.1 Basic idea

When a planet passes between its host star and our line of sight, it
blocks a small fraction of the star's light. This event is called a
**transit**. If we continuously measure the star's brightness over time
(a measurement we call a **light curve**), the transit appears as a
periodic dip.

```
            planet
              o
         _____|_____
        /     |     \       Star
       |      |      |
        \_____↓_____/
              |
         (to Earth)

Brightness
    ^
1.0 ┤────┐         ┌────
    │    │         │
    │    │  dip    │        <- transit
    │    └─────────┘
    └───────────────────> Time
```

For example, if a Jupiter-sized planet transits a Sun-sized star, it
blocks about **1%** of the star's light. An Earth-sized planet around
the same star blocks only **0.008%** — a signal buried in noise.

### 2.2 What we measure

A **light curve** is a time series of flux (brightness) measurements.
For TESS, each measurement is taken every **2 minutes** (short cadence)
or every **30 minutes** (full-frame images).

Key observables from a transit:

| Parameter | What it tells us | How we measure it |
|-----------|-----------------|-------------------|
| **Period** (P) | Orbital period | Time between successive dips |
| **Depth** (delta) | Planet-to-star radius ratio | Fractional brightness decrease: delta = (R_p / R_star)^2 |
| **Duration** (T) | Orbital distance + inclination | Width of the dip |
| **Shape** | Planet vs. binary star | Box-like (planet) vs. V-shaped (binary) |

### 2.3 Why transits are rare

A transit only occurs if the planet's orbit is nearly edge-on to our
line of sight. The probability that a randomly-oriented orbit produces a
transit is approximately:

```
P(transit) ~ R_star / a
```

where `a` is the orbital distance. For Earth around the Sun,
`P ~ 0.5%`. For a hot Jupiter at 0.05 AU, `P ~ 10%`. This is why
large-scale surveys like TESS observe hundreds of thousands of stars —
to find the few percent that are transiting.

---

## 3. The TESS mission

### 3.1 Overview

The **Transiting Exoplanet Survey Satellite** (TESS) is a NASA space
telescope launched in 2018. Its primary mission is to survey the
brightest, nearest stars for transiting exoplanets.

Key facts:

- **Coverage**: Nearly the entire sky, divided into **sectors**
  (24 x 96 degrees each).
- **Observation duration**: Each sector is observed for about **27.4
  days** before the satellite moves to the next one.
- **Cadence**: 2-minute exposures for pre-selected targets (SPOC
  pipeline), 30-minute full-frame images for all stars in the field.
- **Catalog**: Each target has a unique **TIC ID** (TESS Input Catalog
  identifier), e.g. `TIC 150428135` is the star TOI-700.

### 3.2 Data products

TESS data is processed by the **Science Processing Operations Center
(SPOC)** at NASA Ames and archived at **MAST** (Mikulski Archive for
Space Telescopes). The primary data product is a time series of flux
measurements — the light curve.

The light curves we download contain:
- **Time**: in BTJD (Barycentric TESS Julian Date) — the time corrected
  for the light travel time between Earth and the barycentre of the
  Solar System.
- **Flux**: the measured brightness, typically in electrons per second.
- **Flux error**: the estimated uncertainty on each measurement.

### 3.3 Noise sources

The raw light curves contain several sources of noise that can mask or
mimic transit signals:

| Noise source | Timescale | Amplitude | Mitigation |
|-------------|-----------|-----------|------------|
| **Photon noise** | Random (each cadence) | ~100-300 ppm for typical stars | Averaging / binning |
| **Stellar variability** | Hours to days (starspots, rotation) | 0.1%-1% | Savitzky-Golay detrending |
| **Instrumental systematics** | Minutes to hours (pointing jitter, thermal) | ~50-200 ppm | Pipeline corrections |
| **Cosmic rays** | Instantaneous (single cadences) | Large spikes | Sigma-clipping |

**CDPP** (Combined Differential Photometric Precision) is a standard
metric that quantifies how well we can detect transits in a given light
curve. It is measured in parts per million (ppm) and represents the
noise level on the timescale of a typical transit (~13 hours). Lower
CDPP = better sensitivity.

---

## 4. The Box Least Squares (BLS) algorithm

### 4.1 The problem

Given a noisy light curve with ~20,000 data points, find all periodic
box-shaped dips. We don't know the period, the phase (when the first
transit occurs), the duration, or the depth. We must search over all
possible combinations.

### 4.2 The brute-force approach

For each trial period P:
1. **Phase-fold** the light curve: wrap all data points onto a single
   orbit by computing `phase = (time mod P) / P`, placing all transits
   on top of each other.
2. For each trial duration D and epoch t0:
   - Divide the phase-folded data into "in-transit" (inside the box)
     and "out-of-transit" (outside the box).
   - Compute the mean flux in each group.
   - The **depth** is the difference between the out-of-transit and
     in-transit means.
   - The **power** measures how significant this depth is.
3. Record the (P, D, t0) combination with the highest power.

This produces a **BLS periodogram** — a plot of power vs. period. The
period with the highest peak is the best transit candidate.

### 4.3 The BLS power formula

The BLS statistic (Kovacs, Zucker & Mazeh 2002) for a given trial box is:

```
power = (n_in * n_out * delta^2) / n_total
```

where:
- `n_in` = number of data points inside the transit box
- `n_out` = number of data points outside the transit box
- `delta` = depth = mean_out - mean_in
- `n_total` = total number of data points

This formula naturally balances detection significance: a deeper transit
(larger delta) gives more power, but so does having more in-transit
points (which reduces the uncertainty on the depth estimate).

### 4.4 Why it's called "Box Least Squares"

The method fits a **box-shaped model** to the data:
- Flux = 1.0 outside transit
- Flux = 1.0 - depth inside transit

And finds the box parameters (period, phase, duration, depth) that
**minimise the sum of squared residuals** between the model and the data.
This is equivalent to maximising the BLS power defined above.

### 4.5 The computational challenge

A naive implementation tests every combination:

```
n_periods × n_durations × n_epochs × n_data_points
= 10,000 × 6 × 300 × 20,000 = 3.6 × 10^11 operations
```

This is too slow. The binned prefix-sum optimisation (used in both
astropy's C implementation and our Numba version) reduces this to:

```
n_periods × (n_data_points + n_bins × n_durations)
= 10,000 × (20,000 + 300 × 6) ≈ 2 × 10^8 operations
```

A ~1000x speedup, achieved by:
1. **Binning** the phase-folded data into 300 uniform bins (O(n_data)
   per period).
2. Using a **sliding window** over the bins to evaluate all epochs in
   O(n_bins) per duration, instead of O(n_data) per epoch.

---

## 5. Transit validation

### 5.1 The false positive problem

Not every periodic dip is a planet. Common false positives include:

| False positive | Appearance | How to distinguish |
|----------------|------------|-------------------|
| **Eclipsing binary (EB)** | Deep V-shaped dips, period ~1-10 d | Depth > 5%, V-shape metric > 0.5 |
| **Background EB** | Shallow dips (diluted by target star) | Centroid analysis (not in this pipeline) |
| **Stellar variability** | Quasi-periodic dips | Period << rotation period, different shape |
| **Systematic artefacts** | Dips at spacecraft orbit harmonics | Period matches known artefact periods |

### 5.2 The six validation criteria

ExoHunter applies six tests, chosen based on community standards:

**Test 1: Signal-to-noise ratio (SNR >= 7.0)**

The SNR measures how "loud" the transit signal is relative to the noise.
An SNR of 7 is the standard threshold used by the TESS team and the
Kepler mission. Below this, the "signal" is likely just a noise
fluctuation.

```
SNR = depth / sigma_depth
```

**Test 2: Transit depth (0.01% < depth < 5%)**

The depth is related to the planet-to-star radius ratio:

```
depth = (R_planet / R_star)^2
```

- A depth of 0.01% corresponds to an Earth-sized planet around a
  Sun-sized star — at the limit of TESS sensitivity.
- A depth of 5% corresponds to a radius ratio of ~0.22, which is too
  large for a planet (even Jupiter around the Sun gives only ~1%).
  Depths above 5% are almost certainly eclipsing binaries.

**Test 3: Duration consistency (0.1% < duration/period < 25%)**

For a planet on a circular orbit, the transit duration is related to
the orbital period and the stellar radius:

```
T_transit ~ (P / pi) * (R_star / a)
```

where `a` is the orbital semi-major axis (from Kepler's third law). A
transit lasting more than 25% of the period is physically impossible for
a circular orbit. A transit lasting less than 0.1% is likely a
systematic glitch.

**Test 4: Minimum transit count (>= 3)**

We need at least three observed transits to confirm periodicity. Two
dips could be coincidental noise. With TESS's 27.4-day sectors, this
limits detectable periods to roughly P < 9 days per sector (though
multi-sector observations extend this significantly).

**Test 5: V-shape test (metric <= 0.5)**

A genuine planetary transit has a **flat bottom** because the planet
is fully in front of the star for a significant fraction of the transit.
An eclipsing binary produces a **V-shaped** dip because the two stars
have comparable sizes.

```
V-shape metric:
  0.0 = perfect box (planet)
  1.0 = perfect V (eclipsing binary)
  Threshold: <= 0.5

Measured by comparing flux depth at transit center vs. ingress/egress.
```

**Test 6: Harmonic check**

If we detect a period that is exactly 2x, 3x, 0.5x, or 1/3x of another
candidate's period, it's likely a **period alias** — the algorithm found
a harmonic of the true signal rather than the signal itself.

---

## 6. Cross-matching with known catalogs

### 6.1 The TOI catalog

NASA's **TESS Objects of Interest (TOI)** catalog contains all stars
where a transit-like signal has been detected by the TESS team or
community. Each TOI has:

- A **TIC ID** (the star)
- A **TOI number** (e.g. TOI-700.01, TOI-700.02, TOI-700.03 for the
  three planets)
- A **disposition**: PC (planetary candidate), KP (known planet), FP
  (false positive), etc.

The catalog is maintained at **ExoFOP-TESS** and currently contains
~7,900 entries.

### 6.2 Why cross-match?

When ExoHunter detects a transit candidate, we need to know:
- Is this a **known planet** that we re-detected? (validates the pipeline)
- Is this a **known TOI** at a different period? (possible new planet
  in a multi-planet system)
- Is this a **period alias** of a known signal? (not interesting)
- Is this **genuinely new**? (the exciting case!)

### 6.3 The four-tier classification

| Class | Meaning | Implication |
|-------|---------|-------------|
| **NEW_CANDIDATE** | TIC not in any catalog | Potential new exoplanet! |
| **KNOWN_MATCH** | TIC + period match a known TOI | Pipeline validation |
| **KNOWN_TOI** | TIC in catalog, different period | Multi-planet system? |
| **HARMONIC** | Period is a harmonic of a known TOI | Alias, low priority |

---

## 7. Candidate scoring

### 7.1 Motivation

After validation and cross-matching, we may have dozens or hundreds of
candidates. Not all are equally worth investigating. The **priority
score** ranks candidates by how promising they are for follow-up
observation.

### 7.2 The formula

```
score = SNR × v_shape_factor × depth_factor
```

| Factor | Good (= 1.0) | Penalised | Rationale |
|--------|--------------|-----------|-----------|
| `v_shape_factor` | V-shape metric < 0.5 (box-like) | 0.5 if V-shaped | V-shaped dips are likely eclipsing binaries |
| `depth_factor` | Depth < 2% | 0.7 if depth >= 2% | Very deep transits are more likely binaries or blends |

A candidate with SNR=15, box-like transit, and 0.5% depth gets a
score of 15.0. The same candidate with a V-shaped dip gets 7.5.

---

## 8. The TOI-700 system: our test case

### 8.1 Why TOI-700?

TOI-700 (TIC 150428135) is a small, cool M-dwarf star at 31 parsecs
(~100 light-years) from Earth. It hosts at least three confirmed
transiting planets:

| Planet | Period (days) | Radius (R_Earth) | Depth (ppm) | Notes |
|--------|--------------|-------------------|-------------|-------|
| **b** | 9.977 | 1.01 | 580 | Rocky, similar to Earth in size |
| **c** | 16.051 | 2.63 | 780 | Sub-Neptune |
| **d** | 37.426 | 1.14 | 810 | **In the habitable zone!** |

Planet **d** is especially exciting: it is a rocky planet in the
star's habitable zone — the region where liquid water could exist on
the surface. It is one of the few known Earth-sized planets in a
habitable zone found by TESS.

### 8.2 What ExoHunter does with TOI-700

The pipeline's demo mode generates a synthetic light curve that mimics
351 days of TESS observation of TOI-700, with all three planets'
transits superimposed and realistic noise (~300 ppm). The dashboard
allows students to:

1. Select each planet from the candidate dropdown
2. See the transit dip in the light curve panel
3. View the phase-folded data showing the transit shape
4. Verify that the cross-match correctly classifies them as KNOWN_MATCH

---

## 9. Physical constants used in the pipeline

| Constant | Value | Context |
|----------|-------|---------|
| TESS sector duration | 27.4 days | Observation window per sector |
| TESS short cadence | 2 minutes (1.39 × 10^-3 days) | Time resolution |
| Typical CDPP | 100–500 ppm | Noise level for transit detection |
| SNR threshold | 7.0 | Community standard (Kepler/TESS) |
| Max depth (EB cutoff) | 5% | R_p/R_star ~ 0.22 — too large for a planet |
| SG flatten window | 1001 cadences (~33 hours) | Removes stellar variability > ~1 day |
| CDPP transit duration | 13 hours | Standard metric timescale |

---

## 10. Further reading

### Introductory
- Winn, J. N. (2010). "Transits and Occultations." In *Exoplanets*,
  ed. S. Seager, pp. 55-77. University of Arizona Press.
  [arXiv:1001.2010](https://arxiv.org/abs/1001.2010)
- Deming, D. & Seager, S. (2017). "Illusion and Reality in the
  Atmospheres of Exoplanets." *J. Geophys. Res. Planets*, 122, 53-75.

### BLS algorithm
- Kovacs, G., Zucker, S., & Mazeh, T. (2002). "A box-fitting algorithm
  in the search for periodic transits." *Astron. & Astrophys.*, 391,
  369-377. [arXiv:astro-ph/0206099](https://arxiv.org/abs/astro-ph/0206099)

### TESS mission
- Ricker, G. R. et al. (2015). "Transiting Exoplanet Survey Satellite
  (TESS)." *J. Astron. Telesc. Instrum. Syst.*, 1(1), 014003.

### TOI-700 system
- Gilbert, E. A. et al. (2023). "A Second Earth-Sized Planet in the
  Habitable Zone of the M Dwarf, TOI-700." *Astrophysical Journal
  Letters*, 943, L16.

### lightkurve
- Lightkurve Collaboration (2018). "Lightkurve: Kepler and TESS time
  series analysis in Python." *Astrophysics Source Code Library*,
  ascl:1812.013. [docs.lightkurve.org](https://docs.lightkurve.org/)
