#pragma once
// Restored original round-1 calibration (Nov 17), for the original breadboard.
// Source: method1_test/results/method1_session1/calibration.json
// Complete 64-reading median, ~0.9 V reference, 240 MHz.
// Recheck after reassembly; outside 5-20 pF is unvalidated extrapolation.
namespace Method1Calibration {
constexpr double SLOPE = 3.0739224418808453;
constexpr double INTERCEPT = -18.352037372425478;
inline double capacitancePf(double medianUs) {
  return SLOPE * medianUs + INTERCEPT;
}
}
