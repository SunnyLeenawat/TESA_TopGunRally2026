# Track 1: Capacitance Experiments and Coffee Moisture Meter

TESA Top Gun Rally 2026 #20 - Precision Sensing (ESP32 & Sensor)

Status: implementation plan; no measurement results or winning method established yet.

## 1. Goal and approach

Build and evaluate two competition deliverables using an ESP32-WROOM-32 programmed through Arduino IDE:

1. A capacitance meter for unknown ceramic capacitors in the pF range.
2. A coffee moisture meter using the organizer-supplied electrode assembly.

Experiment with all three measurement methods in the technical slides, then compare additional methods that can use the supplied kit. Record raw data, generate graphs, and write a conclusion after each method. An unsuccessful method is still a useful result if its limitations are demonstrated.

Select the best method separately for each deliverable. The best circuit for ceramic capacitors may not be the best predictor of coffee moisture.

Distinguish throughout this document:

- **Competition requirements:** constraints extracted from the supplied rules.
- **Team experiment choices:** the procedures, comparisons, and internal deadlines below.
- **Unconfirmed details:** questions requiring organizer clarification or physical inspection.

## 2. Confirmed competition requirements

| Item | Requirement |
| --- | --- |
| Processor | ESP32; rules allow any version. Team choice: ESP32-WROOM-32. |
| Sensor and components | Use the organizer-supplied sensor and components only. |
| Interface | Serial Port at **9600 baud**. |
| Trigger | One button press and release starts one measurement. A missed or repeated press can invalidate the attempt. |
| Indicator | Onboard LED illuminates for **200 ms** each time a result is sent. |
| Identification | Print the team code and clearly label the physical device with team code and name. |
| Number format | Two decimal places, required unit, and a blank line between measurement blocks. |
| Evaluation | Two unknown samples per contest; five measurements per sample. |
| Response time | Measured from button press until the result appears on screen. Target **less than 3 seconds**. |
| Weighting | Capacitance: 40%; coffee moisture: 55%. The remaining 5% is unspecified in the supplied rules. |
| Submission freeze | No hardware or firmware changes after handover for that contest. |
| Video evidence | Show both the button press and the displayed result. Assign one operator and one camera operator. |

Capacitance output, without quotation marks:

```text
TGR-xx
Capacitance = 10.00 pF

```

Coffee moisture output:

```text
TGR-xx
Moisture Level = 10.00 %

```

Replace `TGR-xx` with the assigned team code before rehearsal. Two displayed decimal places are a formatting requirement, not proof of 0.01 pF accuracy.

The rules require the team to submit its recording through its assigned group. The plan describes this team responsibility; no automated message sending is part of the implementation.

### Supplied hardware and samples

| Component | Quantity / description |
| --- | --- |
| LM393 | 1 dual comparator; inspect the actual marking and package. |
| Potentiometer | 1 x 10 kOhm. |
| Resistors | 1 each: 1 MOhm, 10 kOhm, 2 kOhm. |
| Small capacitors | 1 each: 5 pF, 10 pF, 15 pF, 20 pF. |
| Other capacitor | 1 x 0.1 uF (100 nF). |
| Momentary switch | 1. |
| Other materials | Connecting wires and two stickers. |
| Sensor | Supplied plastic holder and two PCB electrodes, each 5 cm x 10 cm. The holder has three slots; adjacent installed plates have a stated 1 cm gap. |
| Bean samples | Three samples: 70 g, 70 g, and 30 g. Their moisture values must not be assumed from mass or sample identity. |

Available measurement equipment is the kit itself. Do not assume access to a multimeter, scale, oscilloscope, or LCR meter. Request organizer reference-moisture readings and record when they are available.

## 3. Reference capacitor tolerance: +/-5%

The team has confirmed that the supplied reference capacitors have **+/-5% capacitance tolerance**. This is component tolerance, not a meter-error acceptance limit or a correction to the printed scoring formulas.

| Nominal capacitance | Tolerance magnitude | Possible actual capacitance |
| --- | --- | --- |
| 5 pF | 0.25 pF | 4.75-5.25 pF |
| 10 pF | 0.50 pF | 9.50-10.50 pF |
| 15 pF | 0.75 pF | 14.25-15.75 pF |
| 20 pF | 1.00 pF | 19.00-21.00 pF |

Each physical capacitor has an unknown actual value within its tolerance range. Treat that value as fixed across repetitions and shared across all methods that measure it. Averaging repeated readings reduces random measurement noise; it does not remove reference tolerance.

- Describe calibration errors as **deviation from nominal kit capacitance**, not independently verified absolute accuracy.
- Show horizontal +/-5% reference-tolerance bars separately from vertical measurement-repeatability bars.
- Do not label tolerance bounds as standard deviations or confidence intervals.
- Do not force estimates to the nearest nominal capacitor value.
- Agreement with a tolerance interval does not by itself prove that the meter is accurate.
- Do not choose a winner solely because its small nominal-error advantage is indistinguishable from reference uncertainty.

## 4. Measurement methods

| ID | Method | Measurement and purpose | Priority |
| --- | --- | --- | --- |
| M1 | RC charging with LM393 threshold detection | Time the comparator transition after charging starts; characterize comparator and interrupt delays. | Required: first baseline. |
| M2 | RC charging with repeated ADC threshold detection | Sample charging voltage repeatedly and identify threshold crossing; establish whether ADC timing resolves the transient. | Required. |
| M3 | Fixed charging time followed by one ADC reading | Sweep delay and infer capacitance from the resulting voltage. | Required. |
| M4 | GPIO digital-threshold timing | Time a digital-input transition using the RC circuit; calibrate the effective threshold and timing offset. | Kit-only extension. |
| M5 | Multi-delay ADC curve fitting | Collect one reading per fresh discharge/charge cycle at several delays and fit the charging response. An extension of M3. | Kit-only extension. |
| M6 | ESP32 touch peripheral | Record touch counts with fixed acquisition settings; calibrate counts before reporting pF. | Kit-only extension. |
| M7 | LM393 relaxation oscillator | Measure period/frequency and calibrate against the supplied capacitors. | Stretch experiment after M1-M6. |

Complete M1-M3 before spending the remaining time on extensions. Preserve a working competition candidate while investigating alternatives.

### Shared electrical precautions

- Use the same wire arrangement and test fixture across comparable runs. Disconnect unused front ends rather than leaving their input capacitance connected to the sensing node.
- Record the open test-fixture baseline and the empty assembled bean-cup baseline separately. Empty-cup subtraction measures a change from the cup baseline, not total capacitance.
- Keep hands and movable cables away during acquisition. A change in geometry, lead length, electrode position, or front end creates a new circuit revision and requires new calibration.
- Use 3.3 V-compatible GPIO signals and common ground. Check the actual development board and onboard LED before assigning final pins.
- Prefer ADC1 for voltage acquisition. Keep WiFi and Bluetooth disabled for the competition builds.
- Following the organizer guidance relayed by the user, allocate the single 0.1 uF (100 nF) capacitor across the momentary switch, between GPIO26 and GND, to help suppress button bounce. Retain the ESP32 internal button pullup and software debouncing. This capacitor is not allocated to LM393 supply decoupling or the sensing electrodes. For Method 1, use the supplied 2 kOhm resistor as the LM393 output pullup to 3.3 V, separately from the button circuit.
- Never allocate the same supplied resistor or capacitor to two different physical functions simultaneously. Document a complete parts allocation for each circuit.

### M1: comparator threshold and timing

The ordinary LM393 powered from 3.3 V does not support the slide's 2.086 V reference within its specified common-mode range. Start with a potentiometer reference around **1.0 V**, estimated using calibrated ESP32 ADC readings on the relatively low-impedance potentiometer wiper. This is an ADC estimate, not an independent voltage reference.

Inspect the actual comparator variant and its datasheet. Pull its open-collector output up to **3.3 V** using a supplied resistor. An inverting arrangement can capture the actively driven falling edge: reference on `+`, charging capacitor on `-`. Record the reference before each acquisition group, not during the timing-sensitive transient.

For an ideal discharged capacitor charging through a resistor:

```text
Vc(t) = Vcharge * (1 - exp(-t / (R * C)))
C = -t / (R * ln(1 - Vthreshold / Vcharge))
```

Use seconds, ohms, and farads in the equation; convert farads to pF with a factor of 1e12. The shortcut `C = t/R` only applies at an ideal 63.2% threshold.

With 1 MOhm and the nominal 5-20 pF capacitors, ideal RC time constants are only **5-20 us**, before parasitic capacitance. A lower comparator threshold crosses earlier still. Characterize timing resolution, interrupt latency, comparator delay, and baseline offsets; do not assume they are negligible or subtract an invented fixed delay. Calibrate against the supplied references.

### M2 and M3: ADC feasibility

- Record the actual ADC sampling interval and timing configuration. If the first M2 sample is already above threshold, mark the crossing as unresolved.
- ADC sampling loads the node. Repeated reads can change the charging curve; this is part of the experiment.
- Fix ADC attenuation and calibration settings. Do not assume ADC code 4095 equals 3.3 V.
- For M3, initially sweep requested delays of **1, 2, 5, 10, 20, 50, and 100 us**. Characterize achievable timing; requested delay is not automatically the ADC's actual sample time.
- Avoid readings near zero or saturation when inverting the exponential equation. Record these as invalid rather than producing extreme capacitance estimates.
- The slide's 100 us / 100 pF example is illustrative. It is not automatically appropriate for 5-20 pF standards.

### Additional methods

- **M4:** treat the GPIO threshold as an empirical property, not a precision voltage reference.
- **M5:** use a fresh discharge/charge cycle for each delay. If the earliest achievable ADC sample already misses the transient, document this limitation rather than fitting an unsupported curve.
- **M6:** fix touch acquisition settings and keep raw counts. Touch counts are not capacitance units; characterize their conversion or use them as a moisture-calibration input.
- **M7:** validate the schematic, switching thresholds, and component allocation before construction. Use empirical period-to-capacitance calibration; do not apply a 555 equation to an LM393 oscillator.

Future research outside the competition build: CMOS 555 oscillators, dedicated capacitance-to-digital converters, buffered charge-sharing circuits, and AC impedance techniques. These require components or equipment outside the current kit.

## 5. Experiment sequence

### Stage A: board and measurement harness

1. Identify the exact development board, exposed pins, controllable onboard LED, and LED polarity. Escalate the LED requirement to organizers if the board lacks a controllable onboard LED.
2. Record Arduino IDE version, ESP32 core version, board selection, CPU/timing settings, and library versions.
3. Implement reliable press/release debouncing, one-result-per-press behavior, measurement timeouts, and the required Serial output.
4. Provide experiment and competition modes. Confirm that neither holding the button nor releasing it creates an additional measurement.
5. Establish a one-second acquisition budget per press for the initial comparison; record actual acquisition duration and completed reading count.
6. Use bounded acquisition buffers. If capacity limits require a lower sampling cadence, fix and document it before comparison. Flag overflows; do not silently discard readings or print inside timing-sensitive acquisition.

Serial diagnostics are exported after acquisition. Measure competition response time separately through completion of the displayed result, including communication time.

### Stage B: electrical comparison

For each method:

1. Photograph or draw the wiring and record circuit revision, component allocation, settings, baseline, and limitations.
2. Test all four nominal capacitor values.
3. Collect **five button-triggered results per capacitor per session**.
4. Repeat for **three sessions with power cycling**, rotating capacitor order. This gives 60 final press-level results per method, plus buffered internal readings.
5. Preserve raw sub-readings, final estimates, invalid counts, timestamps, and response duration. Internal readings are not independent button presses.
6. Fit a simple calibration using three reference values and evaluate the fourth. Repeat with each value held out.

Hold out all repetitions of one capacitor together; never randomly split its repeated readings across calibration and evaluation. Holding out 5 pF or 20 pF tests extrapolation. After choosing the method and calibration form, fit its deployment coefficients using all four references, while keeping the earlier validation results.

For affine calibration, assess reference-tolerance sensitivity by refitting at reference interval endpoints. Three training capacitors give eight endpoint combinations per holdout; the final four-capacitor fit gives sixteen. These are reference-tolerance sensitivity envelopes, not complete instrument-uncertainty bounds. Use consistent reference assignments when comparing methods.

Produce these figures after every method:

- Raw response versus nominal capacitance, showing reference tolerance and measured spread.
- Calibrated estimate versus nominal capacitance, with an identity line.
- Residuals against nominal capacitance, in pF and optionally as nominal-relative percentages.
- Reading sequence and press-to-press repeatability distribution.
- Session/power-cycle drift, response time, and invalid-reading rate.
- Method-specific plots: threshold-time distributions, voltage-versus-delay curves, touch counts, or period/frequency.

Use clearly identified statistics: for example, vertical bars showing one standard deviation of press-level results and horizontal bars showing component tolerance. Keep unfiltered measurements and state every filtering rule.

Record a conclusion for each method: useful range, calibration, repeatability, speed, failure modes, and whether it advances to bean testing. A method that cannot resolve the supplied references still gets graphs and a documented outcome.

### Stage C: bean measurements and moisture calibration

1. Label the three supplied samples with stable IDs and request their reference moisture information. Do not assume three distinct moisture levels.
2. Where accessible, retain five organizer reference-meter readings and their mean. Repeats describe reference-meter repeatability, not its absolute accuracy. Confirm reference access and timing before depending on it for calibration.
3. Pilot the **30 g sample first** to establish a common fill level all samples can reach. Keep electrode spacing, orientation, fill volume, settling procedure, and any partial electrode coverage fixed and documented.
4. Without a scale, describe this as controlled fill volume, not controlled sample mass. Do not assume a repeatable fill eliminates density variation.
5. Screen every electrically viable method on the same samples. For each method and sample, perform **three independent empty-and-refill cycles**, with **five button-triggered measurements per refill**.
6. Rotate method order and minimize sample exposure. Do not wet, roast, oven-dry, or otherwise modify samples that must be returned.
7. Separate variation within a fixed fill from variation between refills. Record time and ambient conditions when known; leave unavailable temperature/RH fields blank rather than inventing measurements.

When independent reference moisture labels are available, start with a linear model. With three distinct labelled samples, leave-one-sample-out evaluation is exploratory only: each fit uses two independent samples, and endpoint holdouts extrapolate. Do not use high-degree curves or treat repeated readings as additional independent samples.

Without reference labels, compare electrical sample separation and repeatability only. Do not output invented moisture percentages or claim a calibrated moisture meter.

Produce:

- Raw response or apparent capacitance versus reference moisture.
- Predicted versus reference moisture, with identity line.
- Residuals in **moisture percentage points**, clearly separated from relative-percent errors.
- Within-fill and between-refill variation.

Beans are a lossy dielectric, and different excitation methods may report different apparent capacitances. Evaluate their usefulness for moisture prediction rather than requiring identical pF values across methods.

### Stage D: select and freeze competition candidates

Compare candidates using:

| Criterion | Evidence |
| --- | --- |
| Compliance | Supplied components, correct trigger, LED, and output format. |
| Capacitance performance | Held-out deviation from nominal references and tolerance sensitivity. |
| Moisture performance | Sample-level validation against available reference labels, with limitations stated. |
| Repeatability | Variation between presses and between sessions. |
| Packing sensitivity | Between-refill variation for beans. |
| Reliability | Timeouts, invalid readings, reset behavior, and power-cycle stability. |
| Speed | Button-to-display response time. |
| Practicality | Parts count, wiring stability, and ability to reproduce the calibration. |

First require reliable operation and the response-time target. Then compare validation error, repeatability, and practical robustness. Do not select solely by training R-squared or a nominal-error advantage that reference tolerance cannot distinguish.

Choose capacitance and moisture winners separately. Keep a tested fallback firmware version for use before submission. Record the final circuit, coefficients, firmware version, and configuration before handover; do not modify the submitted entry afterward.

## 6. Planned project organization

The following is the target structure, not a claim that these files already exist:

```text
Track1/
|-- AGENTS.md
|-- Planning.md
|-- .gitignore
|-- firmware/
|   |-- README.md
|   `-- moisture_meter/
|       |-- moisture_meter.ino
|       |-- sensor.cpp / sensor.h
|       |-- calibration.cpp / calibration.h
|       |-- measurement_control.cpp / measurement_control.h
|       |-- config.example.h
|       `-- config.h                 # Local configuration; ignored by Git
|-- calibration/
|   |-- README.md
|   |-- requirements.txt
|   |-- capture_serial.py
|   |-- fit_curve.py
|   |-- plot_results.py
|   |-- data/                      # Original measurements and run metadata
|   `-- results/                   # Coefficients, summaries, and plots
`-- docs/
    |-- wiring/
    |-- experiments/
    `-- demo-script.md
```

Keep the primary `.ino` filename identical to its Arduino sketch folder name. Adjacent `.cpp` and `.h` files contain the measurement implementations and shared control logic. Record tested setup and flashing instructions in the firmware README.

Python tools will use NumPy, pandas, Matplotlib, and pyserial, with tested dependency versions recorded when implemented. Close Arduino Serial Monitor before the Python logger opens the same port.

Firmware modes:

- **Experiment:** bounded raw acquisition followed by diagnostic CSV export, including failure status.
- **Competition:** only the required result block for each accepted press. No menus, debug chatter, or CSV headers in measurement output.

Networking is a future platform integration task, not a requirement for these two judged outputs.

### Measurement data contract

Use explicit units and stable IDs. Store bulky settings in run metadata linked by `run_id`; preserve enough information to reproduce every result.

| Category | Required information |
| --- | --- |
| Provenance | Run ID, method ID, firmware version, configuration revision, circuit revision. |
| Grouping | Sample/reference ID, session ID, refill ID where relevant, press ID, raw reading index. |
| Raw observation | Raw value, raw unit, timestamp, acquisition/timing settings. |
| Estimate | `capacitance_pf` or `moisture_pct` when valid; blank otherwise. |
| Reference | Nominal/reference value, unit, source/type; `tolerance_pct = 5` for the four supplied capacitor references. |
| Quality | Status, elapsed time, number of readings, invalid count, overflow flag. |
| Bean context | Fill procedure, electrode geometry, sample identity, reference-meter readings when available. |

Preserve original raw CSVs. Keep invalid records with explicit status rather than replacing them with zero or silently dropping them. Preserve the relationship between internal readings, press-level estimates, refill groups, and sample-level reference labels.

For each method, generate plots and a short experiment report before advancing. The report records the question, circuit/settings, dataset IDs, calibration, results, limitations, and decision.

## 7. Dates and milestones

All dates are in 2026 and all times are Bangkok time. Internal freezes are team targets; submission and sample-return times come from the rules.

| Date/time | Milestone |
| --- | --- |
| 17 November | Receive/inventory components, identify board/LED, build the harness, and begin M1-M3. |
| 18 November, morning | Complete baseline comparisons and feasible extensions; choose and rehearse the capacitance entry. Bean samples are issued on this date. |
| 18 November, 13:00 | Internal capacitance freeze: verify output, label, wiring, calibration, and response time. |
| 18 November, 14:00-15:00 | Official capacitance submission window. |
| 18-19 November | Obtain reference moisture information and compare viable methods on beans. Confirm return/reconfiguration of the submitted board before scheduling this work on it. |
| 19 November, before 16:00 | Complete all required sample measurements and return every supplied bean sample by the official deadline. |
| 20 November, 10:00 | Internal moisture-entry freeze and final rehearsal. |
| 20 November, 11:00-12:00 | Official moisture-meter submission window. |

Time-box optional methods to protect the submission freezes. Once bean samples are returned, use saved data for analysis; do not assume further physical measurements will be possible.

## 8. Verification and acceptance checklist

### Experimental evidence

- [ ] M1-M3 each have raw measurements, plots, and a documented conclusion.
- [ ] M4-M7 are marked tested, unsuccessful, or deferred with a reason.
- [ ] Figures distinguish +/-5% capacitor tolerance from measurement repeatability.
- [ ] Electrical validation holds out entire reference values, including all repeats.
- [ ] Moisture analysis groups by independent sample and does not inflate sample count with repeats.
- [ ] Missing reference moisture values remain missing; no synthetic calibration labels are presented as measurements.
- [ ] Saved CSVs and settings reproduce the comparison plots and calibration.
- [ ] Final method selection records uncertainty and practical limitations.

### Firmware and hardware rehearsal

- [ ] One normal press/release produces exactly one result; bounce, holding, and release do not duplicate it.
- [ ] The team code, measurement label, two decimal places, unit, baud rate, and blank line match the required format.
- [ ] The onboard LED indication lasts 200 ms.
- [ ] Ten consecutive rehearsal measurements follow the two-sample/five-readings format.
- [ ] Button-to-displayed-result time is under 3 seconds for normal valid measurements.
- [ ] Power cycling preserves the intended method and calibration.
- [ ] Timeouts and invalid readings are handled explicitly in experiment mode; clarify allowed competition failure output before freezing firmware.
- [ ] No Serial printing occurs during timing-sensitive acquisition.
- [ ] Numerical checks cover F-to-pF conversion, calibration, tolerance sensitivity, invalid readings, and group-based validation.
- [ ] Wiring, strain relief available within the kit, labels, and the final configuration are checked before handover.
- [ ] The operator and camera operator rehearse capturing the button press and displayed result together.

## 9. Questions to resolve with organizers

- [ ] Confirm how percentage Error and percentage RMSE are normalized. The printed equations multiply by 100 without clearly showing reference-value normalization.
- [ ] Confirm scoring at exact band boundaries and the conflicting response-time limits (12-15 seconds versus more than 13 seconds).
- [ ] Clarify the unassigned 5% of the published overall weighting.
- [ ] Confirm when calibration-sample moisture labels/reference-meter access become available; do not assume access described for judging also applies during preparation.
- [ ] Confirm expected sample fill/quantity during judging and whether the same packing procedure can be used.
- [ ] Confirm when the capacitance entry is returned and may be reconfigured for the moisture contest.
- [ ] Confirm failure-output expectations when a measurement times out or is invalid.
- [ ] Confirm any restrictions on electrode placement/modification and the use of built-in touch sensing or alternative circuits with supplied parts.
- [ ] Confirm publication rules before releasing competition-derived content or choosing a license.

## 10. Repository and publication practice

- Keep local configuration containing private values ignored before adding it to Git; commit only safe examples.
- Preserve original calibration measurements and team-created wiring diagrams/photos.
- Do not commit organizer PDFs, private group invitations, or private event broker details.
- Do not add networking credentials to experiment CSVs or logs.
- Review `git status` and the actual staged content before committing or publishing.
- Check whether `Planning.md` is intentionally ignored before expecting this document to appear in a commit. Its current ignore rule is a separate repository decision.
- Update `AGENTS.md` when implementation changes the actual project structure or introduces commands.
- Licensing remains undecided until ownership and competition publication rules are confirmed.

This document is the current deliverable. Firmware, analysis scripts, wiring, measurements, and generated plots are subsequent implementation work.

## 11. Sources and provenance

Organizer documents were supplied locally by the user and are not copied into the repository:

1. **Track 1 competition rules**, `การแข่งขัน TESA TOP GUN RALLY 2026 Track 1.pdf`, pages 1-5: supplied components, sensor dimensions, outputs, dates, judging, and submission conditions.
2. **Track 1 technical slides**, `การแข่งขัน TESA TOP GUN_20.pdf`, especially pages 8-9 and 16-22: sensor construction, the three example measurement methods, and alternative techniques.
3. **User-confirmed choices:** Arduino IDE, ESP32-WROOM-32, kit-only reference equipment, kit-first experiments, and +/-5% tolerance for the supplied reference capacitors.

Technical references:

- [TI LM393 family datasheet](https://www.ti.com/lit/ds/symlink/lm393.pdf): supply, common-mode limits, open-collector output, and timing limitations; use the section matching the actual component.
- [Espressif Arduino ESP32 ADC API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/adc.html): ADC configuration and calibrated voltage readings.
- [Espressif Arduino ESP32 touch API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/touch.html): touch counts and acquisition settings.
- [Arduino sketch specification](https://docs.arduino.cc/arduino-cli/sketch-specification/): sketch folder naming and supported source files.

The experimental repeat counts, graphs, calibration comparisons, extension priorities, and internal freeze times are team planning decisions, not additional organizer requirements.
