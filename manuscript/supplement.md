# Supplementary material: Target-conditional information horizons in wind-power event forecasting

This supplement belongs to the Renewable Energy manuscript *Different information horizons shape wind-power event predictability*. It replaces the legacy event-representation supplement and documents the current NP005–NP037 evidence package.

## S1. Study object and information boundary

The target begins at $T$ and contains 17 native 15-min power nodes through $T+4\,\mathrm{h}$. Pizhou power is divided by the provider-supplied 87.45-MW farm capacity; Suining and the external process sites use their site-specific training-period q99.5 scales. All sites use a 0.2 normalized excursion rule, with the physical scale declared separately. The four state classes are neither, upward, downward and both. The downward first-passage label has 16 crossing bins (15–240 min) and a 17th no-crossing bin. Power history ends before the issue time. The strict-clock weather package uses a value fixed 24 h before the valid target time. Future onsite wind and ERA5 are retrospective diagnostics or matured labels and do not enter an operating issue-time row.

The administrative reference for Pizhou is 34.3403°N, 118.0068°E and the supplied onsite wind reference height is 140 m. The Suining aggregate contains 14 turbines and retains a row when at least 80% of turbines are valid. Each turbine is scaled by its first chronological training-period q99.5 value. No arbitrary height conversion is applied; the weather products retain their source heights and are interpreted as issued product information.

## S2. Issue-time contracts and chronological support

### S2.1 Pizhou core process-state protocol (NP005/NP008)
- NP005 question: How do fixed-target event risks and matched information gains change as observations become older?
- Pizhou target: 17 native quarter-hour power nodes from T to T+4 h; 0.2 of the provider-supplied 87.45-MW farm capacity; power divided by 87.45 MW; no clipping of recorded support values.
- Leads: 15, 60, 120, 240, 480, 720 min; test cohort: 6,496 windows; arms: power, power_weather, power_weather_wind; model: ExtraTrees 150 trees, depth 16, leaf 32; no test selection.
- Clock/split: Each issue is T-L; onsite measurements end no later than issue minus1min; UTC boundaries 2024-07-08 16:00:00+00:00, 2024-10-20 16:00:00+00:00; target support Identical target windows across all leads/arms; earliest issue has24h split purge, target end inside split.
- Maturity and update: T+4h+1min <= UTC midnight update <= issue; preceding56day target starts; NP008 reuses frozen NP007 rolling predictions and fixes lambda 0.001 before test scoring.
- NP008 features: issue_power, issue_power_squared, GFS_delta/5, JMA_delta/5, GFS_mean/20, GFS_range/10, issue_power*GFS_delta/5, abs_GFS_delta/5, abs_JMA_delta/5, GFS_coarse_excess_TV/10; state classes: Unchanged17node future4h path and four event classes; paired6496test windows; source hashes and freeze receipt are stored in `outputs/next_paper/np008/protocol.json` and `freeze.json`.

### S2.2 Pizhou timing protocol (NP009)
- Question: Does forecast temporal ordering improve arrival-time allocation beyond occurrence risk and background?
- Timing bins: First ordered drawup/down>=0.2capacity within unchanged17-point[T,T+4h] target; bins15..240min and none; occurrence: Frozen NP008 full rolling four-class marginals; same p for every timing method; arms: frequency, context, ordered, sorted; feature counts: {"context": 14, "ordered": 44, "sorted": 44}; lambda selection: One lambda minimizes mean positive-event conditional RPS across2directions*6leads*3covariate arms on validation.
- Arm construction: context uses endpoint/context covariates; ordered uses the ordered NWP interior trajectory; sorted independently sorts the same interior values and preserves endpoint/context features. Fitting: Only matured positive windows;16-time multinomial with mean-loss L2,C=1/(n*lambda); same causal56day rule; smoothing: Predictive mixture with uniform mass16/(n+16); empirical frequency adds1 to each bin; <100positive or<3classes fallback.
- The validation-frozen timing policy selects lambda=0.001 before test scoring; `selection.json` records the frozen policy and validation hash. NP009 is a 16-time multinomial timing model, not the ExtraTrees timing model used by NP025/NP028.
- NP009 uses the frozen NP008 full rolling occurrence marginals. At 15 min, the downward timing denominator is 2,339 windows; frequency conditional RPS/median-MAE are 0.161377/54.215 min and context values are 0.148245/49.348 min.

### S2.3 Pizhou headline absolute scores
| package | lead_min | arm | metric | score | n |
| --- | --- | --- | --- | --- | --- |
| NP008 | 720 | frequency rolling | return Brier | 0.135149 | 6496 |
| NP008 | 720 | full rolling | return Brier | 0.117423 | 6496 |
| NP008 | 720 | frequency rolling | multiclass Brier | 0.706319 | 6496 |
| NP008 | 720 | full rolling | multiclass Brier | 0.572285 | 6496 |
| NP009 | 15 | frequency | conditional timing RPS | 0.161377 | 2339 |
| NP009 | 15 | context | conditional timing RPS | 0.148245 | 2339 |
| NP009 | 15 | frequency | conditional median MAE (min) | 54.215477 | 2339 |
| NP009 | 15 | context | conditional median MAE (min) | 49.348012 | 2339 |

**NP025**: THREE_ARM_INFORMATION_DECOMPOSITION
- Question: Which process objects receive information from issue-time power history and external historical-forecast weather when each source is evaluated alone and together?
- Site and period: suining; 2023-08-31 16:00:00+00:00 to 2024-08-31 15:45:00+00:00.
- Leads: 15, 60, 120, 240, 480, 720 min; target: 17 native 15-min nodes; 0.2 excursion; four process classes and downward first passage.
- Arms: {"power": "16 issue-time power lags plus known target calendar", "weather": "known target calendar plus four weather variables interpolated to 17 target nodes", "joint": "power arm plus weather arm"}.
- Split: Chronological 60/20/20 blocks with four-hour purge; fixed model budgets; no test selection; event model: ExtraTrees 150 trees, depth16, leaf32, max_features=1.0, seed41; timing model: ExtraTrees 100 trees, depth16, leaf16, max_features=1.0, seed43; occurrence held at training frequency.
- Weather source: Open-Meteo Historical Forecast API; GFS global; manifest and monthly response hashes retained; variables: u100_ms, v100_ms, temperature_1000hPa, surface_pressure; complete support: all requested rows in the archived combined file.

**NP028**: STRICT_ISSUE_TIME_PREVIOUS_DAY1_WEATHER
- Question: Does a fixed previous-day1 external GFS package provide process-state information under a strict issue-time contract?
- Site and period: suining; 2023-08-31 16:00:00+00:00 to 2024-08-31 15:45:00+00:00.
- Leads: 15, 60, 120, 240, 480, 720 min; target: 17 native 15-min nodes; 0.2 excursion; four classes and downward first passage.
- Arms: {"power": "16 issue-time power lags plus target calendar", "weather": "target calendar plus previous_day1 u100/v100, surface pressure, and 2m temperature", "joint": "power arm plus previous_day1 weather arm"}.
- Split: Chronological 60/20/20 support-period blocks with four-hour purge; fixed budgets; no test selection; event model: ExtraTrees 150 trees, depth16, leaf32, max_features=1.0, seed41; timing model: ExtraTrees 100 trees, depth16, leaf16, max_features=1.0, seed43; occurrence held at training frequency.
- Weather source: Open-Meteo Previous Runs API; GFS global; all weather values are fixed 24 h before valid time; variables: u100_ms, v100_ms, surface_pressure_previous_day1, temperature_2m_previous_day1; complete support: 2024-02-16 06:00:00+00:00 to 2024-08-31 15:45:00+00:00.

For NP025, the historical-forecast trajectory is the Open-Meteo GFS global historical-forecast package. Each row is selected by the target-valid trajectory and its archived product timestamp; the row-level rule is that all requested fields must be present in the historical package before the forecast row is accepted. The archive does not expose a model release timestamp, so NP025 is reported as a retrospective information decomposition and is not used as the strict real-time release test. NP028 uses the Open-Meteo Previous Runs GFS global package and fixes every weather value 24 h before target valid time. Complete NP028 weather support begins on 2024-02-16. The NP025 and NP028 weather manifests retain monthly query URLs, response hashes, coordinates, units and returned timestamps.

The chronological split is 60/20/20 with a four-hour purge at each boundary. The event model is ExtraTrees with 150 trees, maximum depth 16, minimum leaf size 32, all available features and seed 41. The conditional timing model uses ExtraTrees with 100 trees, depth 16, leaf size 16 and seed 43 on matured downward-event labels; the occurrence probability is held at the training frequency for the conditional timing product. The matched arms use identical windows and budgets within each package.

## S3. Label and score definitions

For a target path $Y_T$, the upward and downward excursions are running-extrema functionals. A downward crossing label $\tau_{T,\downarrow}$ is the first 15-min node at which the downward excursion reaches 0.2 on the declared site-specific normalized scale; the no-crossing label is bin 16. Conditional timing RPS and median arrival-time error are evaluated on the crossing subset, whose denominator is reported in every table. Event-state Brier scores use all test windows.

The unconditional joint first-passage distribution used as a diagnostic is formed from the event probability $p_{\downarrow}=p_{\mathrm{downward}}+p_{\mathrm{both}}$ and the conditional timing distribution $q$:

$$
\tilde q_{0:15}=p_{\downarrow}q_{0:15},\qquad \tilde q_{16}=1-p_{\downarrow}.
$$

The joint first-passage RPS is reported as an accounting score that retains the occurrence and timing components in one no-crossing-inclusive distribution. The main paper keeps conditional RPS as the primary timing object because the route is designed to improve the timing distribution given a process occurrence.

Relative loss reduction is $100(1-L_\mathrm{candidate}/L_\mathrm{reference})$. Intervals are paired seven-day calendar-block bootstrap percentile intervals with 2,000 draws and seed 41; the same target blocks are used for the compared arms. The fitted models are held fixed in these intervals.

## S4. External power-history process-object validation (NP022)

NP022 uses 15-min SCADA from La Haute Borne (4 turbines, 13,912 test windows and 3,966 downward events at 15 min; 2014–2015) and Suining (14 turbines, 6,996 test windows and 2,653 downward events at 15 min; 2023–2024), with a 17-node, four-hour target and the same normalized excursion rule on each site's training-period q99.5 scale. Each site uses 16 power-history lags ending at issue time, known calendar terms, chronological 60/20/20 splits and a four-hour purge. The result validates the occurrence-versus-first-passage distinction for a power-history forecast object; it does not validate the weather-source increment. Suining is reused as the site for NP024–NP030 information-arm experiments, while La Haute Borne supplies the independent external farm check. The exact 15-min timing results are shown below; intervals are paired seven-day calendar-block bootstrap intervals.

| site | lead_minutes | metric | relative_loss_reduction_pct | low | high | windows | valid_windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lahaute | 15 | conditional_rps | 6.914 | 3.694 | 10.149 | 13912 | 3966 |
| lahaute | 15 | conditional_median_mae_minutes | 7.651 | 4.099 | 11.277 | 13912 | 3966 |
| suining | 15 | conditional_rps | 6.625 | 3.476 | 9.058 | 6996 | 2653 |
| suining | 15 | conditional_median_mae_minutes | 7.644 | 4.506 | 10.087 | 6996 | 2653 |

## S5. Three-arm information decomposition (NP025)

The three matched arms are power history plus calendar, weather trajectory plus calendar, and their concatenation. The following tables give absolute scores, denominators and the complete six-lead matrix.

### S5.1 Absolute scores

| package | lead_min | arm | metric | score | n |
| --- | --- | --- | --- | --- | --- |
| np025 | 15 | power | multiclass Brier | 0.575612 | 6996 |
| np025 | 15 | power | return Brier | 0.160101 | 6996 |
| np025 | 15 | power | conditional timing RPS | 0.154844 | 2653 |
| np025 | 15 | power | joint first-passage RPS | 1.703315 | 6996 |
| np025 | 15 | weather | multiclass Brier | 0.623027 | 6996 |
| np025 | 15 | weather | return Brier | 0.174254 | 6996 |
| np025 | 15 | weather | conditional timing RPS | 0.170936 | 2653 |
| np025 | 15 | weather | joint first-passage RPS | 2.252539 | 6996 |
| np025 | 15 | joint | multiclass Brier | 0.561538 | 6996 |
| np025 | 15 | joint | return Brier | 0.164498 | 6996 |
| np025 | 15 | joint | conditional timing RPS | 0.153557 | 2653 |
| np025 | 15 | joint | joint first-passage RPS | 1.669583 | 6996 |
| np025 | 60 | power | multiclass Brier | 0.600090 | 6996 |
| np025 | 60 | power | return Brier | 0.164154 | 6996 |
| np025 | 60 | power | conditional timing RPS | 0.161427 | 2653 |
| np025 | 60 | power | joint first-passage RPS | 1.887056 | 6996 |
| np025 | 60 | weather | multiclass Brier | 0.623339 | 6996 |
| np025 | 60 | weather | return Brier | 0.174356 | 6996 |
| np025 | 60 | weather | conditional timing RPS | 0.170936 | 2653 |
| np025 | 60 | weather | joint first-passage RPS | 2.252990 | 6996 |
| np025 | 60 | joint | multiclass Brier | 0.583377 | 6996 |
| np025 | 60 | joint | return Brier | 0.166401 | 6996 |
| np025 | 60 | joint | conditional timing RPS | 0.160125 | 2653 |
| np025 | 60 | joint | joint first-passage RPS | 1.881277 | 6996 |
| np025 | 120 | power | multiclass Brier | 0.615617 | 6996 |
| np025 | 120 | power | return Brier | 0.166580 | 6996 |
| np025 | 120 | power | conditional timing RPS | 0.164451 | 2653 |
| np025 | 120 | power | joint first-passage RPS | 2.004923 | 6996 |
| np025 | 120 | weather | multiclass Brier | 0.623467 | 6996 |
| np025 | 120 | weather | return Brier | 0.174329 | 6996 |
| np025 | 120 | weather | conditional timing RPS | 0.170936 | 2653 |
| np025 | 120 | weather | joint first-passage RPS | 2.253479 | 6996 |
| np025 | 120 | joint | multiclass Brier | 0.592563 | 6996 |
| np025 | 120 | joint | return Brier | 0.165971 | 6996 |
| np025 | 120 | joint | conditional timing RPS | 0.167480 | 2653 |
| np025 | 120 | joint | joint first-passage RPS | 2.013336 | 6996 |
| np025 | 240 | power | multiclass Brier | 0.644034 | 6996 |
| np025 | 240 | power | return Brier | 0.173388 | 6996 |
| np025 | 240 | power | conditional timing RPS | 0.166670 | 2653 |
| np025 | 240 | power | joint first-passage RPS | 2.145147 | 6996 |
| np025 | 240 | weather | multiclass Brier | 0.623419 | 6996 |
| np025 | 240 | weather | return Brier | 0.174217 | 6996 |
| np025 | 240 | weather | conditional timing RPS | 0.170936 | 2653 |
| np025 | 240 | weather | joint first-passage RPS | 2.251818 | 6996 |
| np025 | 240 | joint | multiclass Brier | 0.610165 | 6996 |
| np025 | 240 | joint | return Brier | 0.172187 | 6996 |
| np025 | 240 | joint | conditional timing RPS | 0.182556 | 2653 |
| np025 | 240 | joint | joint first-passage RPS | 2.166719 | 6996 |
| np025 | 480 | power | multiclass Brier | 0.670330 | 6996 |
| np025 | 480 | power | return Brier | 0.179329 | 6996 |
| np025 | 480 | power | conditional timing RPS | 0.166551 | 2653 |
| np025 | 480 | power | joint first-passage RPS | 2.308316 | 6996 |
| np025 | 480 | weather | multiclass Brier | 0.622916 | 6996 |
| np025 | 480 | weather | return Brier | 0.173840 | 6996 |
| np025 | 480 | weather | conditional timing RPS | 0.170936 | 2653 |
| np025 | 480 | weather | joint first-passage RPS | 2.252441 | 6996 |
| np025 | 480 | joint | multiclass Brier | 0.624551 | 6996 |
| np025 | 480 | joint | return Brier | 0.174619 | 6996 |
| np025 | 480 | joint | conditional timing RPS | 0.182326 | 2653 |
| np025 | 480 | joint | joint first-passage RPS | 2.241706 | 6996 |
| np025 | 720 | power | multiclass Brier | 0.688773 | 6996 |
| np025 | 720 | power | return Brier | 0.181834 | 6996 |
| np025 | 720 | power | conditional timing RPS | 0.169396 | 2653 |
| np025 | 720 | power | joint first-passage RPS | 2.388272 | 6996 |
| np025 | 720 | weather | multiclass Brier | 0.622417 | 6996 |
| np025 | 720 | weather | return Brier | 0.173585 | 6996 |
| np025 | 720 | weather | conditional timing RPS | 0.170936 | 2653 |
| np025 | 720 | weather | joint first-passage RPS | 2.247789 | 6996 |
| np025 | 720 | joint | multiclass Brier | 0.621429 | 6996 |
| np025 | 720 | joint | return Brier | 0.174678 | 6996 |
| np025 | 720 | joint | conditional timing RPS | 0.168204 | 2653 |
| np025 | 720 | joint | joint first-passage RPS | 2.230083 | 6996 |
| np028 | 15 | power | multiclass Brier | 0.619739 | 3772 |
| np028 | 15 | power | return Brier | 0.161253 | 3772 |
| np028 | 15 | power | conditional timing RPS | 0.153839 | 1328 |
| np028 | 15 | power | joint first-passage RPS | 1.716899 | 3772 |
| np028 | 15 | weather | multiclass Brier | 0.677309 | 3772 |
| np028 | 15 | weather | return Brier | 0.159760 | 3772 |
| np028 | 15 | weather | conditional timing RPS | 0.168910 | 1328 |
| np028 | 15 | weather | joint first-passage RPS | 2.327551 | 3772 |
| np028 | 15 | joint | multiclass Brier | 0.583617 | 3772 |
| np028 | 15 | joint | return Brier | 0.154105 | 3772 |
| np028 | 15 | joint | conditional timing RPS | 0.155691 | 1328 |
| np028 | 15 | joint | joint first-passage RPS | 1.714607 | 3772 |
| np028 | 60 | power | multiclass Brier | 0.652818 | 3772 |
| np028 | 60 | power | return Brier | 0.163466 | 3772 |
| np028 | 60 | power | conditional timing RPS | 0.159858 | 1328 |
| np028 | 60 | power | joint first-passage RPS | 1.894041 | 3772 |
| np028 | 60 | weather | multiclass Brier | 0.676935 | 3772 |
| np028 | 60 | weather | return Brier | 0.159343 | 3772 |
| np028 | 60 | weather | conditional timing RPS | 0.168910 | 1328 |
| np028 | 60 | weather | joint first-passage RPS | 2.321699 | 3772 |
| np028 | 60 | joint | multiclass Brier | 0.608165 | 3772 |
| np028 | 60 | joint | return Brier | 0.156432 | 3772 |
| np028 | 60 | joint | conditional timing RPS | 0.163288 | 1328 |
| np028 | 60 | joint | joint first-passage RPS | 1.887661 | 3772 |
| np028 | 120 | power | multiclass Brier | 0.677774 | 3772 |
| np028 | 120 | power | return Brier | 0.166193 | 3772 |
| np028 | 120 | power | conditional timing RPS | 0.162954 | 1328 |
| np028 | 120 | power | joint first-passage RPS | 2.024545 | 3772 |
| np028 | 120 | weather | multiclass Brier | 0.674381 | 3772 |
| np028 | 120 | weather | return Brier | 0.159339 | 3772 |
| np028 | 120 | weather | conditional timing RPS | 0.168910 | 1328 |
| np028 | 120 | weather | joint first-passage RPS | 2.302440 | 3772 |
| np028 | 120 | joint | multiclass Brier | 0.616242 | 3772 |
| np028 | 120 | joint | return Brier | 0.154848 | 3772 |
| np028 | 120 | joint | conditional timing RPS | 0.165522 | 1328 |
| np028 | 120 | joint | joint first-passage RPS | 1.971873 | 3772 |
| np028 | 240 | power | multiclass Brier | 0.700688 | 3772 |
| np028 | 240 | power | return Brier | 0.172616 | 3772 |
| np028 | 240 | power | conditional timing RPS | 0.167666 | 1328 |
| np028 | 240 | power | joint first-passage RPS | 2.184391 | 3772 |
| np028 | 240 | weather | multiclass Brier | 0.670590 | 3772 |
| np028 | 240 | weather | return Brier | 0.159633 | 3772 |
| np028 | 240 | weather | conditional timing RPS | 0.169272 | 1328 |
| np028 | 240 | weather | joint first-passage RPS | 2.287425 | 3772 |
| np028 | 240 | joint | multiclass Brier | 0.632775 | 3772 |
| np028 | 240 | joint | return Brier | 0.156651 | 3772 |
| np028 | 240 | joint | conditional timing RPS | 0.168180 | 1328 |
| np028 | 240 | joint | joint first-passage RPS | 2.053378 | 3772 |
| np028 | 480 | power | multiclass Brier | 0.749283 | 3772 |
| np028 | 480 | power | return Brier | 0.175866 | 3772 |
| np028 | 480 | power | conditional timing RPS | 0.167251 | 1328 |
| np028 | 480 | power | joint first-passage RPS | 2.539761 | 3772 |
| np028 | 480 | weather | multiclass Brier | 0.675134 | 3772 |
| np028 | 480 | weather | return Brier | 0.159812 | 3772 |
| np028 | 480 | weather | conditional timing RPS | 0.168978 | 1328 |
| np028 | 480 | weather | joint first-passage RPS | 2.293023 | 3772 |
| np028 | 480 | joint | multiclass Brier | 0.656977 | 3772 |
| np028 | 480 | joint | return Brier | 0.159748 | 3772 |
| np028 | 480 | joint | conditional timing RPS | 0.168577 | 1328 |
| np028 | 480 | joint | joint first-passage RPS | 2.206822 | 3772 |
| np028 | 720 | power | multiclass Brier | 0.768864 | 3772 |
| np028 | 720 | power | return Brier | 0.177386 | 3772 |
| np028 | 720 | power | conditional timing RPS | 0.161916 | 1328 |
| np028 | 720 | power | joint first-passage RPS | 2.639331 | 3772 |
| np028 | 720 | weather | multiclass Brier | 0.675779 | 3772 |
| np028 | 720 | weather | return Brier | 0.160541 | 3772 |
| np028 | 720 | weather | conditional timing RPS | 0.169256 | 1328 |
| np028 | 720 | weather | joint first-passage RPS | 2.313774 | 3772 |
| np028 | 720 | joint | multiclass Brier | 0.672560 | 3772 |
| np028 | 720 | joint | return Brier | 0.161167 | 3772 |
| np028 | 720 | joint | conditional timing RPS | 0.166304 | 1328 |
| np028 | 720 | joint | joint first-passage RPS | 2.299966 | 3772 |

### S5.2 No-crossing-inclusive joint accounting

`joint_RPS` is the 17-bin score on all windows. `conditional_RPS_positive` is the original positive-event conditional score and uses `down_events` as its denominator; it is not an all-window score.

| package | lead_min | arm | joint_RPS | event_Brier | conditional_RPS_positive | all_windows | down_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np025 | 15 | power | 1.703315 | 0.575612 | 0.154844 | 6996 | 2653 |
| np025 | 15 | weather | 2.252539 | 0.623027 | 0.170936 | 6996 | 2653 |
| np025 | 15 | joint | 1.669583 | 0.561538 | 0.153557 | 6996 | 2653 |
| np025 | 60 | power | 1.887056 | 0.600090 | 0.161427 | 6996 | 2653 |
| np025 | 60 | weather | 2.252990 | 0.623339 | 0.170936 | 6996 | 2653 |
| np025 | 60 | joint | 1.881277 | 0.583377 | 0.160125 | 6996 | 2653 |
| np025 | 120 | power | 2.004923 | 0.615617 | 0.164451 | 6996 | 2653 |
| np025 | 120 | weather | 2.253479 | 0.623467 | 0.170936 | 6996 | 2653 |
| np025 | 120 | joint | 2.013336 | 0.592563 | 0.167480 | 6996 | 2653 |
| np025 | 240 | power | 2.145147 | 0.644034 | 0.166670 | 6996 | 2653 |
| np025 | 240 | weather | 2.251818 | 0.623419 | 0.170936 | 6996 | 2653 |
| np025 | 240 | joint | 2.166719 | 0.610165 | 0.182556 | 6996 | 2653 |
| np025 | 480 | power | 2.308316 | 0.670330 | 0.166551 | 6996 | 2653 |
| np025 | 480 | weather | 2.252441 | 0.622916 | 0.170936 | 6996 | 2653 |
| np025 | 480 | joint | 2.241706 | 0.624551 | 0.182326 | 6996 | 2653 |
| np025 | 720 | power | 2.388272 | 0.688773 | 0.169396 | 6996 | 2653 |
| np025 | 720 | weather | 2.247789 | 0.622417 | 0.170936 | 6996 | 2653 |
| np025 | 720 | joint | 2.230083 | 0.621429 | 0.168204 | 6996 | 2653 |
| np028 | 15 | power | 1.716899 | 0.619739 | 0.153839 | 3772 | 1328 |
| np028 | 15 | weather | 2.327551 | 0.677309 | 0.168910 | 3772 | 1328 |
| np028 | 15 | joint | 1.714607 | 0.583617 | 0.155691 | 3772 | 1328 |
| np028 | 60 | power | 1.894041 | 0.652818 | 0.159858 | 3772 | 1328 |
| np028 | 60 | weather | 2.321699 | 0.676935 | 0.168910 | 3772 | 1328 |
| np028 | 60 | joint | 1.887661 | 0.608165 | 0.163288 | 3772 | 1328 |
| np028 | 120 | power | 2.024545 | 0.677774 | 0.162954 | 3772 | 1328 |
| np028 | 120 | weather | 2.302440 | 0.674381 | 0.168910 | 3772 | 1328 |
| np028 | 120 | joint | 1.971873 | 0.616242 | 0.165522 | 3772 | 1328 |
| np028 | 240 | power | 2.184391 | 0.700688 | 0.167666 | 3772 | 1328 |
| np028 | 240 | weather | 2.287425 | 0.670590 | 0.169272 | 3772 | 1328 |
| np028 | 240 | joint | 2.053378 | 0.632775 | 0.168180 | 3772 | 1328 |
| np028 | 480 | power | 2.539761 | 0.749283 | 0.167251 | 3772 | 1328 |
| np028 | 480 | weather | 2.293023 | 0.675134 | 0.168978 | 3772 | 1328 |
| np028 | 480 | joint | 2.206822 | 0.656977 | 0.168577 | 3772 | 1328 |
| np028 | 720 | power | 2.639331 | 0.768864 | 0.161916 | 3772 | 1328 |
| np028 | 720 | weather | 2.313774 | 0.675779 | 0.169256 | 3772 | 1328 |
| np028 | 720 | joint | 2.299966 | 0.672560 | 0.166304 | 3772 | 1328 |

## S6. Calendar-only information control (NP031)

NP031 fits the same ExtraTrees event and timing budgets on only the four known target-calendar variables extracted from the frozen NP025 and NP028 datasets. It uses the same train/test masks, target rows and seven-day paired bootstrap. The control is deliberately reported against the frequency reference: it shows whether the headline increments require power or weather information beyond calendar seasonality.

| package | lead_min | task | metric | relative_gain_pct | low | high | calendar_only_score | frequency_reference_score | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| np025 | 15 | event | return_brier | -14.1702 | -17.9079 | -8.2785 | 0.2071 | 0.1814 | 6996 |
| np025 | 15 | event | multiclass_brier | -21.6583 | -30.7799 | -9.0857 | 0.8421 | 0.6922 | 6996 |
| np025 | 15 | downward_timing | conditional_rps | -2.9968 | -4.8543 | -1.2232 | 0.1708 | 0.1658 | 6996 |
| np025 | 15 | downward_timing | conditional_median_mae_minutes | -4.3857 | -6.8077 | -1.7052 | 57.5971 | 55.1772 | 6996 |
| np025 | 60 | event | return_brier | -14.0702 | -17.7205 | -8.2810 | 0.2070 | 0.1814 | 6996 |
| np025 | 60 | event | multiclass_brier | -21.0482 | -30.0026 | -8.6542 | 0.8378 | 0.6922 | 6996 |
| np025 | 60 | downward_timing | conditional_rps | -2.9968 | -4.8543 | -1.2232 | 0.1708 | 0.1658 | 6996 |
| np025 | 60 | downward_timing | conditional_median_mae_minutes | -4.3857 | -6.8077 | -1.7052 | 57.5971 | 55.1772 | 6996 |
| np025 | 120 | event | return_brier | -14.3417 | -18.0079 | -8.5330 | 0.2075 | 0.1814 | 6996 |
| np025 | 120 | event | multiclass_brier | -21.1662 | -30.1581 | -8.6800 | 0.8386 | 0.6921 | 6996 |
| np025 | 120 | downward_timing | conditional_rps | -2.9968 | -4.8543 | -1.2232 | 0.1708 | 0.1658 | 6996 |
| np025 | 120 | downward_timing | conditional_median_mae_minutes | -4.3857 | -6.8077 | -1.7052 | 57.5971 | 55.1772 | 6996 |
| np025 | 240 | event | return_brier | -14.4278 | -18.1697 | -8.5476 | 0.2076 | 0.1814 | 6996 |
| np025 | 240 | event | multiclass_brier | -21.0943 | -30.0818 | -8.6728 | 0.8381 | 0.6921 | 6996 |
| np025 | 240 | downward_timing | conditional_rps | -2.9968 | -4.8543 | -1.2232 | 0.1708 | 0.1658 | 6996 |
| np025 | 240 | downward_timing | conditional_median_mae_minutes | -4.3857 | -6.8077 | -1.7052 | 57.5971 | 55.1772 | 6996 |
| np025 | 480 | event | return_brier | -15.0350 | -18.8706 | -9.0295 | 0.2087 | 0.1814 | 6996 |
| np025 | 480 | event | multiclass_brier | -21.6593 | -30.7796 | -9.1520 | 0.8420 | 0.6921 | 6996 |
| np025 | 480 | downward_timing | conditional_rps | -2.9968 | -4.8543 | -1.2232 | 0.1708 | 0.1658 | 6996 |
| np025 | 480 | downward_timing | conditional_median_mae_minutes | -4.3857 | -6.8077 | -1.7052 | 57.5971 | 55.1772 | 6996 |
| np025 | 720 | event | return_brier | -15.2267 | -19.0535 | -9.1325 | 0.2090 | 0.1814 | 6996 |
| np025 | 720 | event | multiclass_brier | -21.4973 | -30.6949 | -8.9407 | 0.8408 | 0.6921 | 6996 |
| np025 | 720 | downward_timing | conditional_rps | -2.9968 | -4.8543 | -1.2232 | 0.1708 | 0.1658 | 6996 |
| np025 | 720 | downward_timing | conditional_median_mae_minutes | -4.3857 | -6.8077 | -1.7052 | 57.5971 | 55.1772 | 6996 |
| np028 | 15 | event | return_brier | -8.5677 | -15.5932 | -1.8888 | 0.1799 | 0.1657 | 3772 |
| np028 | 15 | event | multiclass_brier | -18.7265 | -24.2332 | -13.2029 | 0.8058 | 0.6787 | 3772 |
| np028 | 15 | downward_timing | conditional_rps | -1.3328 | -3.5225 | 2.2674 | 0.1680 | 0.1658 | 3772 |
| np028 | 15 | downward_timing | conditional_median_mae_minutes | -1.7569 | -3.9515 | 1.9211 | 56.2613 | 55.2899 | 3772 |
| np028 | 60 | event | return_brier | -8.5737 | -15.3728 | -1.9854 | 0.1799 | 0.1657 | 3772 |
| np028 | 60 | event | multiclass_brier | -19.6064 | -25.4691 | -14.0056 | 0.8118 | 0.6787 | 3772 |
| np028 | 60 | downward_timing | conditional_rps | -1.3328 | -3.5225 | 2.2674 | 0.1680 | 0.1658 | 3772 |
| np028 | 60 | downward_timing | conditional_median_mae_minutes | -1.7569 | -3.9515 | 1.9211 | 56.2613 | 55.2899 | 3772 |
| np028 | 120 | event | return_brier | -8.4323 | -15.2420 | -1.7417 | 0.1797 | 0.1657 | 3772 |
| np028 | 120 | event | multiclass_brier | -19.7892 | -25.6869 | -14.2221 | 0.8131 | 0.6788 | 3772 |
| np028 | 120 | downward_timing | conditional_rps | -1.3328 | -3.5225 | 2.2674 | 0.1680 | 0.1658 | 3772 |
| np028 | 120 | downward_timing | conditional_median_mae_minutes | -1.7569 | -3.9515 | 1.9211 | 56.2613 | 55.2899 | 3772 |
| np028 | 240 | event | return_brier | -8.2671 | -14.9434 | -1.4807 | 0.1794 | 0.1657 | 3772 |
| np028 | 240 | event | multiclass_brier | -19.5303 | -25.3454 | -14.0686 | 0.8112 | 0.6787 | 3772 |
| np028 | 240 | downward_timing | conditional_rps | -1.9251 | -3.9863 | 1.5465 | 0.1690 | 0.1658 | 3772 |
| np028 | 240 | downward_timing | conditional_median_mae_minutes | -2.2063 | -4.5195 | 1.8440 | 56.5098 | 55.2899 | 3772 |
| np028 | 480 | event | return_brier | -8.3645 | -15.0467 | -1.7860 | 0.1796 | 0.1657 | 3772 |
| np028 | 480 | event | multiclass_brier | -19.1795 | -25.0394 | -13.6208 | 0.8089 | 0.6787 | 3772 |
| np028 | 480 | downward_timing | conditional_rps | -1.6869 | -3.6548 | 1.6676 | 0.1686 | 0.1658 | 3772 |
| np028 | 480 | downward_timing | conditional_median_mae_minutes | -1.9408 | -4.1447 | 1.6884 | 56.3630 | 55.2899 | 3772 |
| np028 | 720 | event | return_brier | -8.5053 | -14.9592 | -2.0278 | 0.1798 | 0.1657 | 3772 |
| np028 | 720 | event | multiclass_brier | -20.3707 | -25.8396 | -14.4917 | 0.8167 | 0.6785 | 3772 |
| np028 | 720 | downward_timing | conditional_rps | -1.4823 | -3.4320 | 1.7714 | 0.1683 | 0.1658 | 3772 |
| np028 | 720 | downward_timing | conditional_median_mae_minutes | -1.9408 | -4.1659 | 1.6178 | 56.3630 | 55.2899 | 3772 |

The calendar-only control has negative event-state relative reductions for both packages and negative conditional timing reductions at the primary leads. The information-arm gains therefore include information beyond the shared calendar columns.

## S7. Target-aligned routing (NP026 and NP029)

NP026 and NP029 are deterministic transformations of frozen test predictions. The route keeps the joint event-state probabilities and assigns the power-history timing probabilities to downward first passage. No model is refit, no test lead is selected, and no test month is used to define the mapping. State identity is checked by maximum absolute probability difference.

| package | lead_minutes | metric | gain (95% CI) % | windows |
| --- | --- | --- | --- | --- |
| np026 | 15 | timing RPS | -0.84 (-3.02, 1.19) | 6996 |
| np026 | 15 | timing median MAE | -1.11 (-3.18, 0.79) | 6996 |
| np026 | 60 | timing RPS | -0.81 (-2.42, 0.75) | 6996 |
| np026 | 60 | timing median MAE | -1.07 (-2.69, 0.58) | 6996 |
| np026 | 120 | timing RPS | 1.81 (-0.18, 3.80) | 6996 |
| np026 | 120 | timing median MAE | 2.09 (0.10, 4.21) | 6996 |
| np026 | 240 | timing RPS | 8.70 (4.96, 12.84) | 6996 |
| np026 | 240 | timing median MAE | 8.01 (5.10, 10.97) | 6996 |
| np026 | 480 | timing RPS | 8.65 (5.55, 12.20) | 6996 |
| np026 | 480 | timing median MAE | 8.79 (5.00, 12.94) | 6996 |
| np026 | 720 | timing RPS | -0.71 (-2.15, 0.52) | 6996 |
| np026 | 720 | timing median MAE | -1.45 (-3.48, 0.66) | 6996 |
| np029 | 15 | timing RPS | 1.19 (-1.89, 3.60) | 3772 |
| np029 | 15 | timing median MAE | 2.43 (-1.49, 5.52) | 3772 |
| np029 | 60 | timing RPS | 2.10 (0.44, 4.15) | 3772 |
| np029 | 60 | timing median MAE | 2.52 (-0.56, 5.46) | 3772 |
| np029 | 120 | timing RPS | 1.55 (0.45, 2.23) | 3772 |
| np029 | 120 | timing median MAE | 1.64 (0.31, 3.14) | 3772 |
| np029 | 240 | timing RPS | 0.31 (-1.46, 1.61) | 3772 |
| np029 | 240 | timing median MAE | -0.22 (-2.04, 0.99) | 3772 |
| np029 | 480 | timing RPS | 0.79 (-1.39, 4.03) | 3772 |
| np029 | 480 | timing median MAE | 1.27 (-1.46, 4.96) | 3772 |
| np029 | 720 | timing RPS | 2.64 (-0.71, 6.42) | 3772 |
| np029 | 720 | timing median MAE | 2.39 (-1.15, 6.95) | 3772 |

The complete matrices are retained in `outputs/next_paper/np026/routing_summary.csv` and `outputs/next_paper/np029/routing_summary.csv`. The primary route endpoints are 240 and 480 min for the historical package and 60 and 120 min for the strict-clock package; all six leads remain in the supplement.

## S8. Monthly stability audits (NP027 and NP030)

NP027 recomputes the frozen NP026 route separately for UTC June, July and August 2024. NP030 recomputes the strict-clock route for July and August 2024. No month, model or lead is selected during the audit. The complete monthly tables, including mixed signs at secondary leads, are retained in the corresponding output directories and figure source manifests. The main text reports the pre-specified supported endpoints; the audit is an auxiliary stability check.

## S9. Reproducibility and source traceability

The current evidence package is generated from the following frozen output directories: `np005`, `np007`, `np008`, `np009`, `np013`, `np014`, `np016`, `np017`, `np018`, `np019`, `np020`, `np022`, `np024`, `np025`, `np026`, `np027`, `np028`, `np029`, `np030`, `np031_calendar_control`, `np032_np033`, `np034_weather_ablation`, `np035_calibration`, `np036_joint_timing_baseline` and `np037_threshold_sensitivity`. Each directory contains a protocol or manifest, test predictions or summary, and an independent verification record. NP005/NP007/NP008/NP009 bind the Pizhou cohort, feature definitions, frozen occurrence predictions, timing predictions and validation-frozen lambda. NP013/NP014/NP016–NP020 are retained supporting range/calibration/fusion outputs referenced by the main text and their protocols remain in the corresponding output directories. NP025 and NP028 bind their scripts, SCADA source, weather file and weather manifest by SHA-256. NP026 and NP029 bind their routing scripts to the frozen source prediction manifests. NP031 records the dataset source and the four-column calendar extraction. NP032/NP033 record the seed and calendar robustness extensions; NP034/NP035 record the weather-variable ablation and calibration diagnostics.

NP037 adds the threshold reconstruction check under outputs/next_paper/np037_threshold_sensitivity, and the exact NP022 external timing rows and NP014 range/path rows are reproduced in S4 and S15.

Raw private SCADA is not redistributed. The minimum reproducibility index consists of the protocol JSON, support CSV, prediction summary CSV, verification JSON and source-hash manifest in each listed output directory; these files are sufficient to replay the reported score arithmetic after the private source is replaced by the permitted derived arrays. The released replication package will include public weather files or source URLs, feature/label arrays after privacy review, prediction summaries, source hashes and the exact scripts used to produce the tables. All timestamp handling is UTC.

## S10. Secondary matrices and scope

The full six-lead, arm-by-metric relative-loss matrices, all monthly rows, support counts and negative or mixed increments remain part of the machine-readable output directories. The main text uses the pre-specified positive endpoints to explain the information-to-target rule; these secondary matrices provide the complete audit trail for the interpretation.

## S11. Seed and month robustness extensions (NP032/NP033)

NP032 refits the NP025 and NP028 event and timing models at seeds 41, 42 and 43 for the 15-, 240- and 720-min endpoints. The central state increments remain stable: at 720 min, joint-versus-power multiclass Brier gains average 9.56% (SD 0.48%) for NP025 and 12.61% (SD 0.57%) for NP028. The full seed matrix is retained below.

| index | package | lead_minutes | arm | metric | mean | std | min | max | count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | np025 | 15 | joint_added_to_power | multiclass_brier | 2.356 | 0.153 | 2.180 | 2.445 | 3 |
| 1 | np025 | 15 | power_added_to_weather | conditional_rps | 8.989 | 0.417 | 8.693 | 9.466 | 3 |
| 2 | np025 | 15 | weather_added_to_power | multiclass_brier | -8.481 | 0.227 | -8.685 | -8.237 | 3 |
| 3 | np025 | 240 | joint_added_to_power | multiclass_brier | 5.179 | 0.073 | 5.116 | 5.259 | 3 |
| 4 | np025 | 240 | power_added_to_weather | conditional_rps | 2.038 | 0.566 | 1.563 | 2.665 | 3 |
| 5 | np025 | 240 | weather_added_to_power | multiclass_brier | 2.971 | 0.253 | 2.699 | 3.201 | 3 |
| 6 | np025 | 720 | joint_added_to_power | multiclass_brier | 9.555 | 0.476 | 9.008 | 9.880 | 3 |
| 7 | np025 | 720 | power_added_to_weather | conditional_rps | 0.280 | 0.879 | -0.487 | 1.239 | 3 |
| 8 | np025 | 720 | weather_added_to_power | multiclass_brier | 9.368 | 0.544 | 8.742 | 9.728 | 3 |
| 9 | np028 | 15 | joint_added_to_power | multiclass_brier | 5.738 | 0.140 | 5.576 | 5.828 | 3 |
| 10 | np028 | 15 | power_added_to_weather | conditional_rps | 9.106 | 0.356 | 8.728 | 9.435 | 3 |
| 11 | np028 | 15 | weather_added_to_power | multiclass_brier | -9.254 | 0.930 | -10.166 | -8.308 | 3 |
| 12 | np028 | 240 | joint_added_to_power | multiclass_brier | 10.196 | 0.535 | 9.692 | 10.757 | 3 |
| 13 | np028 | 240 | power_added_to_weather | conditional_rps | 0.939 | 0.639 | 0.473 | 1.667 | 3 |
| 14 | np028 | 240 | weather_added_to_power | multiclass_brier | 4.241 | 0.801 | 3.414 | 5.014 | 3 |
| 15 | np028 | 720 | joint_added_to_power | multiclass_brier | 12.612 | 0.569 | 12.091 | 13.219 | 3 |
| 16 | np028 | 720 | power_added_to_weather | conditional_rps | 3.986 | 0.122 | 3.878 | 4.118 | 3 |
| 17 | np028 | 720 | weather_added_to_power | multiclass_brier | 12.640 | 0.528 | 12.107 | 13.162 | 3 |

NP033 recalculates frozen information increments by calendar month for all six leads. The summary below reports mean, SD, minimum, maximum and count across supported months; the complete row matrix is stored in `outputs/next_paper/np032_np033/monthly_information.csv`.

| index | package | arm | task | metric | mean | std | min | max | count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | np025 | joint | event | multiclass_brier | 14.345 | 4.090 | 7.249 | 23.241 | 18 |
| 1 | np025 | power | event | multiclass_brier | 8.260 | 6.511 | -3.355 | 17.568 | 18 |
| 2 | np025 | power_added_to_weather | downward_timing | conditional_rps | 3.767 | 3.597 | -2.325 | 10.950 | 18 |
| 3 | np025 | weather | event | multiclass_brier | 10.845 | 2.657 | 7.154 | 13.128 | 18 |
| 4 | np028 | joint | event | multiclass_brier | 6.049 | 5.177 | -1.360 | 16.904 | 12 |
| 5 | np028 | power | event | multiclass_brier | -3.012 | 7.615 | -13.701 | 10.788 | 12 |
| 6 | np028 | power_added_to_weather | downward_timing | conditional_rps | 3.293 | 3.692 | -2.884 | 9.631 | 12 |
| 7 | np028 | weather | event | multiclass_brier | -0.054 | 1.307 | -1.532 | 1.865 | 12 |

## S12. Calibration diagnostics (NP035)

NP035 bins frozen state probabilities and the probability of a downward arrival within 60 min into ten probability bins. It reports bin count, mean predicted probability, observed frequency and absolute calibration gap for NP025/NP028, all leads and all arms. These diagnostics separate information value from calibration: a lower forecast loss can coexist with a calibration gap, and the main paper therefore uses proper losses as its primary comparison while retaining reliability rows here.

| index | package | target | mean | max | count |
| --- | --- | --- | --- | --- | --- |
| 0 | np025 | downward | 0.0892 | 0.9058 | 171 |
| 1 | np025 | downward_arrival_le_60min | 0.1891 | 0.6274 | 121 |
| 2 | np025 | return | 0.1060 | 0.4032 | 109 |
| 3 | np028 | downward | 0.1124 | 0.4749 | 169 |
| 4 | np028 | downward_arrival_le_60min | 0.1850 | 0.6097 | 112 |
| 5 | np028 | return | 0.0821 | 0.4398 | 121 |

The representative bin-level rows below expose the calibration calculation for the 15-min and 12-h joint arms; the complete reliability table is stored in `outputs/next_paper/np035_calibration/reliability.csv`.

| package | lead_minutes | target | bin_low | bin_high | n | mean_pred | obs_rate | abs_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| np025 | 15 | return | 0.0000 | 0.1000 | 3069 | 0.0392 | 0.0997 | 0.0605 |
| np025 | 15 | return | 0.1000 | 0.2000 | 1387 | 0.1501 | 0.2408 | 0.0907 |
| np025 | 15 | return | 0.2000 | 0.3000 | 1178 | 0.2488 | 0.3438 | 0.0950 |
| np025 | 15 | return | 0.3000 | 0.4000 | 872 | 0.3464 | 0.3945 | 0.0481 |
| np025 | 15 | return | 0.4000 | 0.5000 | 425 | 0.4347 | 0.4188 | 0.0159 |
| np025 | 15 | return | 0.5000 | 0.6000 | 45 | 0.5361 | 0.8889 | 0.3528 |
| np025 | 15 | return | 0.6000 | 0.7000 | 20 | 0.6328 | 0.7500 | 0.1172 |
| np025 | 15 | downward | 0.0000 | 0.1000 | 2364 | 0.0402 | 0.0943 | 0.0541 |
| np025 | 15 | downward | 0.1000 | 0.2000 | 850 | 0.1480 | 0.2282 | 0.0803 |
| np025 | 15 | downward | 0.2000 | 0.3000 | 623 | 0.2496 | 0.2921 | 0.0425 |
| np025 | 15 | downward | 0.3000 | 0.4000 | 573 | 0.3494 | 0.3333 | 0.0161 |
| np025 | 15 | downward | 0.4000 | 0.5000 | 518 | 0.4490 | 0.4517 | 0.0027 |
| np025 | 15 | downward | 0.5000 | 0.6000 | 473 | 0.5493 | 0.6152 | 0.0659 |
| np025 | 15 | downward | 0.6000 | 0.7000 | 623 | 0.6477 | 0.7368 | 0.0891 |
| np025 | 15 | downward | 0.7000 | 0.8000 | 546 | 0.7487 | 0.8626 | 0.1140 |
| np025 | 15 | downward | 0.8000 | 0.9000 | 357 | 0.8428 | 0.9524 | 0.1096 |
| np025 | 15 | downward | 0.9000 | 1.0000 | 69 | 0.9239 | 0.9855 | 0.0616 |
| np025 | 720 | return | 0.0000 | 0.1000 | 2900 | 0.0460 | 0.1155 | 0.0695 |
| np025 | 720 | return | 0.1000 | 0.2000 | 1694 | 0.1484 | 0.2928 | 0.1444 |
| np025 | 720 | return | 0.2000 | 0.3000 | 1375 | 0.2440 | 0.3105 | 0.0665 |
| np025 | 720 | return | 0.3000 | 0.4000 | 684 | 0.3434 | 0.3246 | 0.0189 |
| np025 | 720 | return | 0.4000 | 0.5000 | 285 | 0.4443 | 0.4246 | 0.0198 |
| np025 | 720 | return | 0.5000 | 0.6000 | 55 | 0.5423 | 0.3455 | 0.1968 |
| np025 | 720 | return | 0.6000 | 0.7000 | 3 | 0.6079 | 0.6667 | 0.0588 |
| np025 | 720 | downward | 0.0000 | 0.1000 | 1263 | 0.0618 | 0.1108 | 0.0490 |
| np025 | 720 | downward | 0.1000 | 0.2000 | 1075 | 0.1480 | 0.2381 | 0.0901 |
| np025 | 720 | downward | 0.2000 | 0.3000 | 1109 | 0.2495 | 0.3517 | 0.1021 |
| np025 | 720 | downward | 0.3000 | 0.4000 | 885 | 0.3483 | 0.4395 | 0.0912 |
| np025 | 720 | downward | 0.4000 | 0.5000 | 838 | 0.4475 | 0.4403 | 0.0071 |
| np025 | 720 | downward | 0.5000 | 0.6000 | 840 | 0.5494 | 0.5131 | 0.0363 |
| np025 | 720 | downward | 0.6000 | 0.7000 | 635 | 0.6490 | 0.6709 | 0.0219 |
| np025 | 720 | downward | 0.7000 | 0.8000 | 258 | 0.7416 | 0.7403 | 0.0013 |
| np025 | 720 | downward | 0.8000 | 0.9000 | 93 | 0.8409 | 0.6559 | 0.1849 |
| np028 | 15 | return | 0.0000 | 0.1000 | 1700 | 0.0473 | 0.1047 | 0.0574 |
| np028 | 15 | return | 0.1000 | 0.2000 | 892 | 0.1394 | 0.2085 | 0.0691 |
| np028 | 15 | return | 0.2000 | 0.3000 | 465 | 0.2458 | 0.3011 | 0.0553 |
| np028 | 15 | return | 0.3000 | 0.4000 | 368 | 0.3491 | 0.3234 | 0.0257 |
| np028 | 15 | return | 0.4000 | 0.5000 | 253 | 0.4422 | 0.4862 | 0.0440 |
| np028 | 15 | return | 0.5000 | 0.6000 | 87 | 0.5362 | 0.4253 | 0.1109 |
| np028 | 15 | return | 0.6000 | 0.7000 | 7 | 0.6547 | 0.7143 | 0.0596 |
| np028 | 15 | downward | 0.0000 | 0.1000 | 944 | 0.0399 | 0.0985 | 0.0586 |
| np028 | 15 | downward | 0.1000 | 0.2000 | 720 | 0.1457 | 0.1708 | 0.0252 |
| np028 | 15 | downward | 0.2000 | 0.3000 | 341 | 0.2413 | 0.2053 | 0.0361 |
| np028 | 15 | downward | 0.3000 | 0.4000 | 191 | 0.3445 | 0.2618 | 0.0828 |
| np028 | 15 | downward | 0.4000 | 0.5000 | 210 | 0.4473 | 0.3857 | 0.0616 |
| np028 | 15 | downward | 0.5000 | 0.6000 | 202 | 0.5518 | 0.4109 | 0.1409 |
| np028 | 15 | downward | 0.6000 | 0.7000 | 269 | 0.6503 | 0.5911 | 0.0593 |
| np028 | 15 | downward | 0.7000 | 0.8000 | 393 | 0.7517 | 0.6412 | 0.1105 |
| np028 | 15 | downward | 0.8000 | 0.9000 | 401 | 0.8450 | 0.7930 | 0.0520 |
| np028 | 15 | downward | 0.9000 | 1.0000 | 101 | 0.9193 | 0.9802 | 0.0609 |
| np028 | 720 | return | 0.0000 | 0.1000 | 1072 | 0.0556 | 0.1474 | 0.0918 |
| np028 | 720 | return | 0.1000 | 0.2000 | 1299 | 0.1484 | 0.1440 | 0.0044 |
| np028 | 720 | return | 0.2000 | 0.3000 | 647 | 0.2439 | 0.3091 | 0.0652 |
| np028 | 720 | return | 0.3000 | 0.4000 | 390 | 0.3471 | 0.2667 | 0.0804 |
| np028 | 720 | return | 0.4000 | 0.5000 | 296 | 0.4402 | 0.3446 | 0.0956 |
| np028 | 720 | return | 0.5000 | 0.6000 | 63 | 0.5327 | 0.5079 | 0.0247 |
| np028 | 720 | return | 0.6000 | 0.7000 | 5 | 0.6090 | 1.0000 | 0.3910 |
| np028 | 720 | downward | 0.0000 | 0.1000 | 381 | 0.0599 | 0.2598 | 0.2000 |
| np028 | 720 | downward | 0.1000 | 0.2000 | 551 | 0.1469 | 0.1706 | 0.0237 |
| np028 | 720 | downward | 0.2000 | 0.3000 | 472 | 0.2497 | 0.2606 | 0.0109 |
| np028 | 720 | downward | 0.3000 | 0.4000 | 480 | 0.3539 | 0.3979 | 0.0440 |
| np028 | 720 | downward | 0.4000 | 0.5000 | 480 | 0.4551 | 0.2896 | 0.1655 |
| np028 | 720 | downward | 0.5000 | 0.6000 | 615 | 0.5454 | 0.4618 | 0.0836 |
| np028 | 720 | downward | 0.6000 | 0.7000 | 420 | 0.6460 | 0.4810 | 0.1650 |
| np028 | 720 | downward | 0.7000 | 0.8000 | 280 | 0.7470 | 0.5214 | 0.2256 |
| np028 | 720 | downward | 0.8000 | 0.9000 | 93 | 0.8280 | 0.5376 | 0.2904 |
| np025 | 15 | downward_arrival_le_60min | 0.0000 | 0.1000 | 4820 | 0.0371 | 0.0193 | 0.0178 |
| np025 | 15 | downward_arrival_le_60min | 0.1000 | 0.2000 | 761 | 0.1453 | 0.1091 | 0.0362 |
| np025 | 15 | downward_arrival_le_60min | 0.2000 | 0.3000 | 503 | 0.2475 | 0.2445 | 0.0029 |
| np025 | 15 | downward_arrival_le_60min | 0.3000 | 0.4000 | 443 | 0.3487 | 0.3183 | 0.0304 |
| np025 | 15 | downward_arrival_le_60min | 0.4000 | 0.5000 | 381 | 0.4415 | 0.3963 | 0.0452 |
| np025 | 15 | downward_arrival_le_60min | 0.5000 | 0.6000 | 71 | 0.5378 | 0.6620 | 0.1241 |
| np025 | 15 | downward_arrival_le_60min | 0.6000 | 0.7000 | 17 | 0.6206 | 0.5882 | 0.0324 |
| np025 | 720 | downward_arrival_le_60min | 0.0000 | 0.1000 | 376 | 0.0781 | 0.0160 | 0.0621 |
| np025 | 720 | downward_arrival_le_60min | 0.1000 | 0.2000 | 3942 | 0.1598 | 0.0944 | 0.0655 |
| np025 | 720 | downward_arrival_le_60min | 0.2000 | 0.3000 | 1910 | 0.2372 | 0.0974 | 0.1398 |
| np025 | 720 | downward_arrival_le_60min | 0.3000 | 0.4000 | 563 | 0.3413 | 0.1012 | 0.2401 |
| np025 | 720 | downward_arrival_le_60min | 0.4000 | 0.5000 | 164 | 0.4345 | 0.1585 | 0.2760 |
| np025 | 720 | downward_arrival_le_60min | 0.5000 | 0.6000 | 40 | 0.5369 | 0.0250 | 0.5119 |
| np025 | 720 | downward_arrival_le_60min | 0.6000 | 0.7000 | 1 | 0.6144 | 0.0000 | 0.6144 |
| np028 | 15 | downward_arrival_le_60min | 0.0000 | 0.1000 | 2487 | 0.0603 | 0.0197 | 0.0406 |
| np028 | 15 | downward_arrival_le_60min | 0.1000 | 0.2000 | 404 | 0.1424 | 0.1064 | 0.0360 |
| np028 | 15 | downward_arrival_le_60min | 0.2000 | 0.3000 | 232 | 0.2502 | 0.1422 | 0.1079 |
| np028 | 15 | downward_arrival_le_60min | 0.3000 | 0.4000 | 261 | 0.3519 | 0.2222 | 0.1297 |
| np028 | 15 | downward_arrival_le_60min | 0.4000 | 0.5000 | 265 | 0.4493 | 0.2604 | 0.1890 |
| np028 | 15 | downward_arrival_le_60min | 0.5000 | 0.6000 | 113 | 0.5279 | 0.4159 | 0.1120 |
| np028 | 15 | downward_arrival_le_60min | 0.6000 | 0.7000 | 10 | 0.6153 | 0.5000 | 0.1153 |
| np028 | 720 | downward_arrival_le_60min | 0.0000 | 0.1000 | 81 | 0.0886 | 0.0741 | 0.0145 |
| np028 | 720 | downward_arrival_le_60min | 0.1000 | 0.2000 | 1598 | 0.1549 | 0.0582 | 0.0967 |
| np028 | 720 | downward_arrival_le_60min | 0.2000 | 0.3000 | 1570 | 0.2440 | 0.1057 | 0.1383 |
| np028 | 720 | downward_arrival_le_60min | 0.3000 | 0.4000 | 308 | 0.3348 | 0.0552 | 0.2796 |
| np028 | 720 | downward_arrival_le_60min | 0.4000 | 0.5000 | 183 | 0.4540 | 0.0874 | 0.3666 |
| np028 | 720 | downward_arrival_le_60min | 0.5000 | 0.6000 | 32 | 0.5294 | 0.1875 | 0.3419 |

The protocol and verification receipts bind the reliability rows to the frozen NP025/NP028 predictions.

## S13. Joint first-passage baseline and route falsification (NP036)

NP036 fits a direct 17-class first-passage model with the joint feature arm under the same chronological windows and ExtraTrees budget. The route is evaluated both against this direct joint model and against the constructed joint distribution from the original joint event/timing heads. The direct baseline is a stronger end-to-end comparator; the constructed comparator isolates the value of replacing the conditional timing head while preserving the state head. The full six-lead matrix is stored in `outputs/next_paper/np036_joint_timing_baseline/joint_timing_baseline.csv`.

| package | route_package | lead_minutes | metric | route_score | direct_joint_score | route_gain_vs_direct_pct | windows | reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| np025 | np026 | 15 | joint_first_passage_RPS | 1.6759 | 1.6598 | -0.9672 | 6996 | nan |
| np025 | np026 | 15 | joint_first_passage_RPS | 1.6759 | 1.6696 | -0.3758 | 6996 | constructed_joint |
| np025 | np026 | 60 | joint_first_passage_RPS | 1.8884 | 1.8753 | -0.6992 | 6996 | nan |
| np025 | np026 | 60 | joint_first_passage_RPS | 1.8884 | 1.8813 | -0.3790 | 6996 | constructed_joint |
| np025 | np026 | 120 | joint_first_passage_RPS | 2.0093 | 2.0046 | -0.2344 | 6996 | nan |
| np025 | np026 | 120 | joint_first_passage_RPS | 2.0093 | 2.0133 | 0.1982 | 6996 | constructed_joint |
| np025 | np026 | 240 | joint_first_passage_RPS | 2.1434 | 2.1346 | -0.4094 | 6996 | nan |
| np025 | np026 | 240 | joint_first_passage_RPS | 2.1434 | 2.1667 | 1.0768 | 6996 | constructed_joint |
| np025 | np026 | 480 | joint_first_passage_RPS | 2.2158 | 2.1857 | -1.3754 | 6996 | nan |
| np025 | np026 | 480 | joint_first_passage_RPS | 2.2158 | 2.2417 | 1.1575 | 6996 | constructed_joint |
| np025 | np026 | 720 | joint_first_passage_RPS | 2.2317 | 2.2079 | -1.0744 | 6996 | nan |
| np025 | np026 | 720 | joint_first_passage_RPS | 2.2317 | 2.2301 | -0.0707 | 6996 | constructed_joint |
| np028 | np029 | 15 | joint_first_passage_RPS | 1.6883 | 1.7125 | 1.4172 | 3772 | nan |
| np028 | np029 | 15 | joint_first_passage_RPS | 1.6883 | 1.7146 | 1.5361 | 3772 | constructed_joint |
| np028 | np029 | 60 | joint_first_passage_RPS | 1.8668 | 1.9018 | 1.8379 | 3772 | nan |
| np028 | np029 | 60 | joint_first_passage_RPS | 1.8668 | 1.8877 | 1.1045 | 3772 | constructed_joint |
| np028 | np029 | 120 | joint_first_passage_RPS | 1.9600 | 1.9947 | 1.7399 | 3772 | nan |
| np028 | np029 | 120 | joint_first_passage_RPS | 1.9600 | 1.9719 | 0.6016 | 3772 | constructed_joint |
| np028 | np029 | 240 | joint_first_passage_RPS | 2.0436 | 2.1734 | 5.9699 | 3772 | nan |
| np028 | np029 | 240 | joint_first_passage_RPS | 2.0436 | 2.0534 | 0.4747 | 3772 | constructed_joint |
| np028 | np029 | 480 | joint_first_passage_RPS | 2.1900 | 2.4662 | 11.2023 | 3772 | nan |
| np028 | np029 | 480 | joint_first_passage_RPS | 2.1900 | 2.2068 | 0.7645 | 3772 | constructed_joint |
| np028 | np029 | 720 | joint_first_passage_RPS | 2.2857 | 2.5637 | 10.8416 | 3772 | nan |
| np028 | np029 | 720 | joint_first_passage_RPS | 2.2857 | 2.3000 | 0.6192 | 3772 | constructed_joint |

## S14. Weather-variable ablation (NP034)

NP034 compares calendar-only, thermodynamic-only, wind-vector-only and full-weather feature sets at 15, 240 and 720 min under the same tree budgets. The wind-vector channel supplies the positive state-risk contribution in both packages; thermodynamic-only features remain below the frequency reference in the tested ablation.

| package | lead_minutes | arm | task | metric | absolute_score | relative_vs_frequency_pct | windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| np025 | 15 | calendar | event | multiclass_brier | 0.84207 | -21.65830 | 6996 |
| np025 | 15 | calendar | downward_timing | conditional_rps | 0.17029 | nan | 2653 |
| np025 | 15 | wind_vector | event | multiclass_brier | 0.62075 | 10.31707 | 6996 |
| np025 | 15 | wind_vector | downward_timing | conditional_rps | 0.17694 | nan | 2653 |
| np025 | 15 | thermodynamic | event | multiclass_brier | 0.77281 | -11.65267 | 6996 |
| np025 | 15 | thermodynamic | downward_timing | conditional_rps | 0.18526 | nan | 2653 |
| np025 | 15 | full_weather | event | multiclass_brier | 0.62303 | 9.98776 | 6996 |
| np025 | 15 | full_weather | downward_timing | conditional_rps | 0.16972 | nan | 2653 |
| np025 | 240 | calendar | event | multiclass_brier | 0.83813 | -21.09426 | 6996 |
| np025 | 240 | calendar | downward_timing | conditional_rps | 0.17029 | nan | 2653 |
| np025 | 240 | wind_vector | event | multiclass_brier | 0.62015 | 10.39998 | 6996 |
| np025 | 240 | wind_vector | downward_timing | conditional_rps | 0.17694 | nan | 2653 |
| np025 | 240 | thermodynamic | event | multiclass_brier | 0.75693 | -9.36245 | 6996 |
| np025 | 240 | thermodynamic | downward_timing | conditional_rps | 0.18526 | nan | 2653 |
| np025 | 240 | full_weather | event | multiclass_brier | 0.62342 | 9.92732 | 6996 |
| np025 | 240 | full_weather | downward_timing | conditional_rps | 0.16972 | nan | 2653 |
| np025 | 720 | calendar | event | multiclass_brier | 0.84084 | -21.49735 | 6996 |
| np025 | 720 | calendar | downward_timing | conditional_rps | 0.17029 | nan | 2653 |
| np025 | 720 | wind_vector | event | multiclass_brier | 0.62064 | 10.32070 | 6996 |
| np025 | 720 | wind_vector | downward_timing | conditional_rps | 0.17694 | nan | 2653 |
| np025 | 720 | thermodynamic | event | multiclass_brier | 0.74896 | -8.21986 | 6996 |
| np025 | 720 | thermodynamic | downward_timing | conditional_rps | 0.18526 | nan | 2653 |
| np025 | 720 | full_weather | event | multiclass_brier | 0.62242 | 10.06424 | 6996 |
| np025 | 720 | full_weather | downward_timing | conditional_rps | 0.16972 | nan | 2653 |
| np028 | 15 | calendar | event | multiclass_brier | 0.80579 | -18.72647 | 3772 |
| np028 | 15 | calendar | downward_timing | conditional_rps | 0.16768 | nan | 1328 |
| np028 | 15 | wind_vector | event | multiclass_brier | 0.64427 | 5.07253 | 3772 |
| np028 | 15 | wind_vector | downward_timing | conditional_rps | 0.16869 | nan | 1328 |
| np028 | 15 | thermodynamic | event | multiclass_brier | 0.85863 | -26.51232 | 3772 |
| np028 | 15 | thermodynamic | downward_timing | conditional_rps | 0.17020 | nan | 1328 |
| np028 | 15 | full_weather | event | multiclass_brier | 0.67731 | 0.20442 | 3772 |
| np028 | 15 | full_weather | downward_timing | conditional_rps | 0.16936 | nan | 1328 |
| np028 | 240 | calendar | event | multiclass_brier | 0.81121 | -19.53029 | 3772 |
| np028 | 240 | calendar | downward_timing | conditional_rps | 0.16751 | nan | 1328 |
| np028 | 240 | wind_vector | event | multiclass_brier | 0.64138 | 5.49444 | 3772 |
| np028 | 240 | wind_vector | downward_timing | conditional_rps | 0.16900 | nan | 1328 |
| np028 | 240 | thermodynamic | event | multiclass_brier | 0.86281 | -27.13345 | 3772 |
| np028 | 240 | thermodynamic | downward_timing | conditional_rps | 0.16953 | nan | 1328 |
| np028 | 240 | full_weather | event | multiclass_brier | 0.67059 | 1.19019 | 3772 |
| np028 | 240 | full_weather | downward_timing | conditional_rps | 0.16975 | nan | 1328 |
| np028 | 720 | calendar | event | multiclass_brier | 0.81675 | -20.37071 | 3772 |
| np028 | 720 | calendar | downward_timing | conditional_rps | 0.16737 | nan | 1328 |
| np028 | 720 | wind_vector | event | multiclass_brier | 0.63631 | 6.22230 | 3772 |
| np028 | 720 | wind_vector | downward_timing | conditional_rps | 0.16919 | nan | 1328 |
| np028 | 720 | thermodynamic | event | multiclass_brier | 0.86625 | -27.66617 | 3772 |
| np028 | 720 | thermodynamic | downward_timing | conditional_rps | 0.16990 | nan | 1328 |
| np028 | 720 | full_weather | event | multiclass_brier | 0.67578 | 0.40500 | 3772 |
| np028 | 720 | full_weather | downward_timing | conditional_rps | 0.16913 | nan | 1328 |

## S15. Range and path supporting results (NP014)

NP014 supplies a separate amplitude and path-order channel. The first two rows compare the direct range arm with the old conditional range reference; the third compares the path-order assignment with the same range marginal after order shuffling. The table gives the exact headline values used in the main text; the full range, coverage and event-value matrices remain in the NP014 output package.

| lead_min | task | candidate | reference | gain_pct | CI_low | CI_high | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | range CRPS | 0.551 | 0.677 | 18.570 | 12.940 | 24.240 | 6059 |
| 720 | range CRPS | 0.564 | 0.676 | 16.610 | 11.610 | 22.050 | 6059 |
| 720 | return Brier; shuffled order | 0.121 | 0.122 | 1.120 | 0.200 | 2.130 | 6496 |

## S16. Excursion-threshold sensitivity (NP037)

NP037 reconstructs the Suining aggregate from the archived SCADA and reruns the fixed power, weather and joint feature contracts at thresholds 0.15, 0.20 and 0.25. The 0.20 labels reproduce the frozen NP025/NP028 labels exactly for every tested lead and package. At the two off-design thresholds, the weather-only state point gain remains positive at all six package–lead combinations; the timing table reports the added value of power history relative to weather.

### S16.1 Weather state increment at thresholds 0.15 and 0.25

| package | threshold | lead_minutes | relative_vs_frequency_pct | windows |
| --- | --- | --- | --- | --- |
| np025 | 0.150 | 15 | 9.769 | 6996 |
| np025 | 0.250 | 15 | 9.924 | 6996 |
| np025 | 0.150 | 240 | 9.884 | 6996 |
| np025 | 0.250 | 240 | 9.947 | 6996 |
| np025 | 0.150 | 720 | 9.704 | 6996 |
| np025 | 0.250 | 720 | 9.947 | 6996 |
| np028 | 0.150 | 15 | 7.452 | 3772 |
| np028 | 0.250 | 15 | 4.022 | 3772 |
| np028 | 0.150 | 240 | 7.031 | 3772 |
| np028 | 0.250 | 240 | 4.508 | 3772 |
| np028 | 0.150 | 720 | 7.171 | 3772 |
| np028 | 0.250 | 720 | 4.579 | 3772 |

### S16.2 Power-history timing increment across thresholds

| package | threshold | lead_minutes | power_added_to_weather_pct |
| --- | --- | --- | --- |
| np025 | 0.150 | 15 | 10.754 |
| np025 | 0.150 | 240 | 0.833 |
| np025 | 0.150 | 720 | -0.131 |
| np025 | 0.200 | 15 | 9.414 |
| np025 | 0.200 | 240 | 2.496 |
| np025 | 0.200 | 720 | 0.901 |
| np025 | 0.250 | 15 | 6.387 |
| np025 | 0.250 | 240 | 0.940 |
| np025 | 0.250 | 720 | 1.164 |
| np028 | 0.150 | 15 | 10.749 |
| np028 | 0.150 | 240 | 1.583 |
| np028 | 0.150 | 720 | 1.404 |
| np028 | 0.200 | 15 | 8.923 |
| np028 | 0.200 | 240 | 0.949 |
| np028 | 0.200 | 720 | 4.337 |
| np028 | 0.250 | 15 | 4.515 |
| np028 | 0.250 | 240 | -0.990 |
| np028 | 0.250 | 720 | 2.868 |

## S17. Mathematical synthesis and operational interface

The forecast family used in the paper is

$$
\mathcal{Y}_{q,T}=\Phi_q(Y_T),\qquad
\widehat F_{q,h}^{\,a}=f_q(X_{T-h}^{\,a},h),\qquad
\mathcal{L}_{q,h}^{\,a}=\mathbb{E}\,S_q(\widehat F_{q,h}^{\,a},\mathcal{Y}_{q,T}).
$$

The empirical information value is

$$
\widehat{\mathrm{IV}}_q(A\rightarrow B;h)
=100\frac{\widehat{\mathcal{L}}_{q,h}^{\,A}-\widehat{\mathcal{L}}_{q,h}^{\,B}}
{\widehat{\mathcal{L}}_{q,h}^{\,A}}.
$$

For a downward occurrence probability $p_{\downarrow}$ and a conditional first-passage distribution $q_\tau$, the no-crossing-inclusive accounting distribution is

$$
\widetilde q_k=p_{\downarrow}q_{\tau,k}\quad(k=0,\ldots,15),\qquad
\widetilde q_{16}=1-p_{\downarrow}.
$$

The target-aligned route preserves the joint state vector and replaces only the conditional timing channel. Therefore, the constructed joint-score change is attributable to the timing-channel replacement under the fixed state identity; NP036 supplies the direct 17-bin joint comparison. The operational interface uses the return-event probability for a state alert and the conditional timing distribution for timing allocation. NP023 varies the alert threshold for a stated false-negative/false-positive loss while keeping forecast probabilities frozen.

## S18. Capacity-matched information controls (NP038)

NP038 compares the original feature arms with compact weather summaries, a fixed projection, timestamp-matched noise, and power-plus-weather/noise arms at matched dimensions. Weather summaries use the four calendar columns plus mean, standard deviation, first node and last node for each issued weather variable. The projection is standardized on training rows before a fixed QR map; the noise arm is generated from timestamp-hashed random seeds. The complete 432-row score table, 648-row seed table, feature maps, labels, predictions and tree-complexity records are released in the NP038 directory.

| package | comparison | min | max | mean |
| --- | --- | --- | --- | --- |
| np025 | power_weather_summary40_vs_power_noise40 | 5.670 | 11.329 | 7.718 |
| np025 | weather_summary20_vs_noise20 | 25.574 | 27.807 | 26.768 |
| np028 | power_weather_summary40_vs_power_noise40 | 6.825 | 9.206 | 8.074 |
| np028 | weather_summary20_vs_noise20 | 14.978 | 15.915 | 15.430 |

The rows below provide the primary matched state-score comparisons; interval bounds are paired seven-day blocks with 2,000 draws.

| package | lead_minutes | comparison | candidate_score | reference_score | reduction_pct | reduction_low | reduction_high | valid_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| np025 | 15 | weather_summary20_vs_noise20 | 0.608 | 0.842 | 27.807 | 22.441 | 31.847 | 6996 |
| np025 | 15 | power_weather_summary40_vs_power_noise40 | 0.545 | 0.580 | 6.069 | 3.591 | 9.066 | 6996 |
| np025 | 60 | weather_summary20_vs_noise20 | 0.607 | 0.834 | 27.185 | 21.769 | 31.244 | 6996 |
| np025 | 60 | power_weather_summary40_vs_power_noise40 | 0.569 | 0.603 | 5.670 | 2.942 | 9.033 | 6996 |
| np025 | 120 | weather_summary20_vs_noise20 | 0.607 | 0.835 | 27.317 | 21.871 | 31.417 | 6996 |
| np025 | 120 | power_weather_summary40_vs_power_noise40 | 0.581 | 0.620 | 6.231 | 2.905 | 10.265 | 6996 |
| np025 | 240 | weather_summary20_vs_noise20 | 0.607 | 0.829 | 26.715 | 21.336 | 30.785 | 6996 |
| np025 | 240 | power_weather_summary40_vs_power_noise40 | 0.603 | 0.651 | 7.328 | 2.489 | 12.908 | 6996 |
| np025 | 480 | weather_summary20_vs_noise20 | 0.608 | 0.821 | 26.009 | 20.613 | 30.101 | 6996 |
| np025 | 480 | power_weather_summary40_vs_power_noise40 | 0.616 | 0.681 | 9.680 | 5.163 | 14.472 | 6996 |
| np025 | 720 | weather_summary20_vs_noise20 | 0.609 | 0.818 | 25.574 | 20.274 | 29.715 | 6996 |
| np025 | 720 | power_weather_summary40_vs_power_noise40 | 0.617 | 0.696 | 11.329 | 6.419 | 16.398 | 6996 |
| np028 | 15 | weather_summary20_vs_noise20 | 0.685 | 0.806 | 14.978 | 11.168 | 20.640 | 3772 |
| np028 | 15 | power_weather_summary40_vs_power_noise40 | 0.588 | 0.631 | 6.825 | 2.667 | 13.466 | 3772 |
| np028 | 60 | weather_summary20_vs_noise20 | 0.686 | 0.811 | 15.349 | 11.543 | 21.271 | 3772 |
| np028 | 60 | power_weather_summary40_vs_power_noise40 | 0.614 | 0.665 | 7.763 | 3.004 | 14.659 | 3772 |
| np028 | 120 | weather_summary20_vs_noise20 | 0.689 | 0.815 | 15.461 | 11.583 | 21.101 | 3772 |
| np028 | 120 | power_weather_summary40_vs_power_noise40 | 0.634 | 0.694 | 8.595 | 4.545 | 14.055 | 3772 |
| np028 | 240 | weather_summary20_vs_noise20 | 0.685 | 0.812 | 15.656 | 11.768 | 21.649 | 3772 |
| np028 | 240 | power_weather_summary40_vs_power_noise40 | 0.654 | 0.718 | 8.856 | 5.632 | 13.574 | 3772 |
| np028 | 480 | weather_summary20_vs_noise20 | 0.688 | 0.811 | 15.223 | 11.333 | 20.916 | 3772 |
| np028 | 480 | power_weather_summary40_vs_power_noise40 | 0.711 | 0.766 | 7.198 | 5.121 | 10.130 | 3772 |
| np028 | 720 | weather_summary20_vs_noise20 | 0.684 | 0.813 | 15.915 | 12.211 | 21.106 | 3772 |
| np028 | 720 | power_weather_summary40_vs_power_noise40 | 0.705 | 0.777 | 9.206 | 6.972 | 12.829 | 3772 |

NP038 also retains projection and timing comparisons whose signs vary by package or lead. These rows remain part of the release and are not promoted to the main narrative.

## S19. All-lead route uncertainty and endpoint policy (NP039/NP040)

NP039 recomputes route, constructed-joint and direct-joint 17-bin products with a shared UTC 7-day block universe, 2,000 draws and fixed models. Conditional timing rows use the downward-event mask; state identity is checked against the frozen joint event head. NP040 first averages the six lead-wise ratios within each bootstrap draw and then takes the interval, avoiding interval averaging.

| package | reference | metric | equal_weight_mean_gain_pct | global_low | global_high | min_lead_gain_pct | max_lead_gain_pct | min_valid_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| np025 | constructed_joint | conditional_median_mae_minutes | 2.548 | 0.794 | 4.290 | -1.448 | 8.785 | 2653 |
| np025 | constructed_joint | conditional_rps | 2.790 | 1.011 | 4.891 | -0.838 | 8.702 | 2653 |
| np025 | constructed_joint | joint_first_passage_RPS | 0.272 | -0.163 | 0.743 | -0.379 | 1.158 | 6996 |
| np025 | direct_joint | conditional_median_mae_minutes | -0.645 | -1.721 | 0.415 | -1.830 | 1.371 | 2653 |
| np025 | direct_joint | conditional_rps | -0.177 | -1.290 | 0.920 | -1.986 | 2.185 | 2653 |
| np025 | direct_joint | joint_first_passage_RPS | -0.764 | -1.542 | 0.194 | -1.375 | -0.234 | 6996 |
| np028 | constructed_joint | conditional_median_mae_minutes | 1.709 | 0.601 | 2.460 | -0.221 | 2.518 | 1328 |
| np028 | constructed_joint | conditional_rps | 1.453 | 0.761 | 1.982 | 0.306 | 2.638 | 1328 |
| np028 | constructed_joint | joint_first_passage_RPS | 0.888 | -0.230 | 1.821 | 0.475 | 1.536 | 3772 |
| np028 | direct_joint | conditional_median_mae_minutes | 2.724 | -0.124 | 5.812 | 0.755 | 5.113 | 1328 |
| np028 | direct_joint | conditional_rps | 2.471 | 0.021 | 5.288 | 0.902 | 4.555 | 1328 |
| np028 | direct_joint | joint_first_passage_RPS | 5.824 | 1.015 | 11.595 | 1.417 | 11.202 | 3772 |

The endpoint policy makes all six leads primary. The 240/480-min archived and 60/120-min strict patterns are descriptive local peaks, not test-selected endpoints.

## S20. All-lead decision sensitivity and NP023 comparison (NP041)

NP041 selects alert thresholds only on validation probabilities and applies them once to the test cohort. The full six-lead table includes no-alert, always-alert, validation-frequency and validation-constant policy costs, alert rates, saved loss arrays and paired intervals. The central forecast-product result is the full-rolling cost reduction relative to no alert; the two-lead NP023 comparison is retained separately because NP014/NP019/NP020 are available only at 15 and 720 min.

| lead_minutes | cost_ratio | value | low | high | threshold | windows |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 2.000 | 2.842 | 0.660 | 5.300 | 0.530 | 6496 |
| 15 | 5.000 | 41.407 | 27.516 | 52.286 | 0.105 | 6496 |
| 15 | 10.000 | 34.481 | 20.896 | 48.386 | 0.105 | 6496 |
| 60 | 2.000 | 0.674 | -0.457 | 2.017 | 0.560 | 6496 |
| 60 | 5.000 | 37.360 | 25.950 | 46.547 | 0.185 | 6496 |
| 60 | 10.000 | 32.796 | 22.328 | 43.360 | 0.080 | 6496 |
| 120 | 2.000 | 0.337 | -0.264 | 1.293 | 0.600 | 6496 |
| 120 | 5.000 | 39.422 | 25.268 | 48.972 | 0.130 | 6496 |
| 120 | 10.000 | 33.510 | 21.999 | 44.811 | 0.090 | 6496 |
| 240 | 2.000 | 0.000 | 0.000 | 0.000 | 0.690 | 6496 |
| 240 | 5.000 | 36.705 | 23.839 | 44.818 | 0.160 | 6496 |
| 240 | 10.000 | 32.649 | 23.423 | 42.943 | 0.075 | 6496 |
| 480 | 2.000 | 2.119 | -0.946 | 5.508 | 0.470 | 6496 |
| 480 | 5.000 | 42.466 | 32.045 | 50.026 | 0.150 | 6496 |
| 480 | 10.000 | 29.626 | 19.803 | 40.029 | 0.075 | 6496 |
| 720 | 2.000 | 1.156 | -0.941 | 4.824 | 0.495 | 6496 |
| 720 | 5.000 | 41.985 | 32.622 | 48.650 | 0.140 | 6496 |
| 720 | 10.000 | 30.542 | 21.113 | 40.674 | 0.080 | 6496 |

### S20.1 NP023 fusion comparison

| lead_minutes | cost_ratio | candidate | reference | relative_cost_reduction_pct | low | high |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 2.000 | np013_selected | np014_selected | 1.820 | -11.368 | 13.994 |
| 15 | 2.000 | np014_selected | np014_selected | 0.000 | 0.000 | 0.000 |
| 15 | 2.000 | np019_selected | np014_selected | -0.590 | -4.324 | 2.723 |
| 15 | 2.000 | np020_selected | np014_selected | -0.590 | -4.324 | 2.723 |
| 15 | 5.000 | np013_selected | np014_selected | -9.849 | -19.448 | -1.383 |
| 15 | 5.000 | np014_selected | np014_selected | 0.000 | 0.000 | 0.000 |
| 15 | 5.000 | np019_selected | np014_selected | -2.664 | -6.823 | 1.854 |
| 15 | 5.000 | np020_selected | np014_selected | -2.664 | -6.823 | 1.854 |
| 15 | 10.000 | np013_selected | np014_selected | -29.851 | -53.701 | -13.898 |
| 15 | 10.000 | np014_selected | np014_selected | 0.000 | 0.000 | 0.000 |
| 15 | 10.000 | np019_selected | np014_selected | 1.640 | -4.749 | 8.859 |
| 15 | 10.000 | np020_selected | np014_selected | 1.640 | -4.749 | 8.859 |
| 720 | 2.000 | np013_selected | np014_selected | -8.264 | -19.219 | -0.791 |
| 720 | 2.000 | np014_selected | np014_selected | 0.000 | 0.000 | 0.000 |
| 720 | 2.000 | np019_selected | np014_selected | 0.875 | -3.468 | 5.921 |
| 720 | 2.000 | np020_selected | np014_selected | 0.875 | -3.468 | 5.921 |
| 720 | 5.000 | np013_selected | np014_selected | -19.403 | -30.324 | -10.886 |
| 720 | 5.000 | np014_selected | np014_selected | 0.000 | 0.000 | 0.000 |
| 720 | 5.000 | np019_selected | np014_selected | 3.402 | -1.503 | 7.947 |
| 720 | 5.000 | np020_selected | np014_selected | 3.402 | -1.503 | 7.947 |
| 720 | 10.000 | np013_selected | np014_selected | -12.334 | -26.038 | 0.067 |
| 720 | 10.000 | np014_selected | np014_selected | 0.000 | 0.000 | 0.000 |
| 720 | 10.000 | np019_selected | np014_selected | -0.211 | -5.054 | 4.787 |
| 720 | 10.000 | np020_selected | np014_selected | -0.211 | -5.054 | 4.787 |

## S21. Literature gap and anonymous reproduction

The verified literature matrix separates NWP ensemble ramps, joint time-distribution ramps, timing/intensity scoring, weather-regime conditioning and probabilistic calibration. The full comparison and official metadata are in temp/literature_gap_verified.md and literature_matrix_30plus.md. The public repository contains the processed aggregate/features/labels, frozen predictions, protocols, score tables, figure manifests and verification scripts. Raw SCADA, turbine-level coordinates, operational-status records and credentials are excluded. A reader can clone the repository anonymously and run the verification scripts without a GitHub account.
