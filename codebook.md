# Codebook: Variable Definitions

## Source Data

**2014 National Health Interview Survey (NHIS)**
- Sample Adult file (`samadult.dat`): Health, tinnitus, noise exposure, sleep
- Person file (`personsx.dat`): Education

## Merge Keys

| Variable | File | Positions | Description |
|----------|------|-----------|-------------|
| HHX | Both | 7–12 | Household serial number |
| FMX | Both | 16–17 | Family serial number |
| FPX | Both | 18–19 | Person number (within family) |

## Survey Design Variables

| Variable | Positions | Description |
|----------|-----------|-------------|
| WTFA_SA | 27–32 | Final annual Sample Adult weight |
| STRAT_P | 34–36 | Pseudo-stratum for variance estimation |
| PSU_P | 37–38 | Pseudo-PSU for variance estimation |

## Outcomes

### Primary: Tinnitus-Related Sleep Disturbance (HRTINSLP)
- **Source**: samadult, position 359
- **Question**: "During the past 12 months, have you been bothered by ringing, roaring, or buzzing in your ears or head when going to sleep?"
- **Universe**: Adults reporting bothersome tinnitus (HRTIN = 1)
- **Coding**: 1 = Yes → 1; 2 = No → 0; 7/8/9 → missing

### Secondary 1: Tinnitus Problem Severity (HRTINPRO)
- **Source**: samadult, position 360
- **Coding**: 3/4/5 (moderate/big/very big problem) → 1; 1/2 (no/small problem) → 0

### Secondary 2: Short Sleep (ASISLEEP)
- **Source**: samadult, positions 1057–1058
- **Question**: Average hours of sleep per 24-hour period
- **Coding**: <7 hours → 1 (short sleep); ≥7 hours → 0; >24 or 97/98/99 → missing

## Exposure Variables

### Occupational Noise (composite)
- **HRWKVLNS** (pos 386): Ever had job with very loud noise exposure
- **HRWKLNS** (pos 387): Ever had job with loud noise exposure
- **Coding**: Either = 1 → occupational noise; both = 2 → no occupational noise

### Leisure Noise (HRLSVLNS)
- **Source**: position 398
- **Coding**: 1 (very loud noise ≥10 times/year outside work) → 1; 2 → 0

### Four-Level Exposure Group (derived)
- 0 = No major noise exposure
- 1 = Occupational noise only
- 2 = Leisure noise only
- 3 = Both occupational and leisure noise

### Hearing Protection at Work (derived)
- **HRWKVLP1** (pos 391): Protection for very loud noise, past 12 months
- **HRWKLP1** (pos 396): Protection for loud noise, past 12 months
- **Coding**: 1 (always) or 2 (usually) → regular (1); 3/4/5 → irregular/none (0)

## Covariates

### Age (AGE_P)
- **Source**: samadult, positions 48–49
- **Coding**: Continuous, ≥18

### Sex (SEX)
- **Source**: samadult, position 39
- **Coding**: 1 = Male → 0; 2 = Female → 1

### Race/Ethnicity (derived from HISPAN_I + RACERPI2)
- **HISPAN_I** (pos 40–41): 0–11 = Hispanic; 12 = Not Hispanic
- **RACERPI2** (pos 42–43): 1 = White; 2 = Black; 3+ = Other
- **Coding**: 0 = NH White (ref); 1 = NH Black; 2 = NH Other; 3 = Hispanic

### Education (EDUC1, from Person file)
- **Source**: personsx, positions 785–786
- **Coding**: 0 = Bachelor+ (ref, codes 18–21); 1 = <HS (codes 0–14); 2 = HS graduate (code 15); 3 = Some college/Associate (codes 16–17); 96/97/98/99 → missing

### Smoking Status (SMKSTAT2)
- **Source**: samadult, position 748
- **Coding**: 0 = Never (ref, code 4); 1 = Former (code 3); 2 = Current (codes 1–2)

### Hearing Difficulty (AHEARST2)
- **Source**: samadult, position 296
- **Question**: Hearing without hearing aid
- **Coding**: 0 = Good hearing (codes 1–2); 1 = Any difficulty (codes 3–6); 7/8/9 → missing

### Firearm Exposure (HRFIRE)
- **Source**: samadult, position 379
- **Coding**: 1 = Ever used firearms → 1; 2 = No → 0

## Missing Value Conventions

| Code | Meaning | Applied to |
|------|---------|------------|
| 7 | Refused | Single-digit variables |
| 8 | Not ascertained | Single-digit variables |
| 9 | Don't know | Single-digit variables |
| 97/98/99 | Refused/NA/DK | Two-digit variables |
| 96 | Child under 5 | EDUC1 |

## Analytic Samples

- **S1 (Tinnitus subsample)**: Adults ≥18 with tinnitus, valid HRTINSLP, valid exposure, education, hearing difficulty, and survey design variables (n = 4,292)
- **S2 (Full adult sample)**: Adults ≥18 with valid ASISLEEP, valid exposure, education, hearing difficulty, and survey design variables (n = 33,507)
