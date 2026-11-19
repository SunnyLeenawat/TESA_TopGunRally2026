#pragma once
#include <math.h>

namespace MoistureCalibration {
// No invented coefficients: fill from measured reference coffee data.
// Feature = measured median charging time - fixed empty-box baseline (us).
constexpr bool READY = false;
constexpr double EMPTY_US = NAN;
constexpr double SLOPE_PERCENT_PER_US = NAN;
constexpr double INTERCEPT_PERCENT = NAN;
constexpr double MIN_DELTA_US = NAN;
constexpr double MAX_DELTA_US = NAN;

inline bool predict(double medianUs, double &moisture) {
  moisture = NAN;
  if (!READY || !isfinite(medianUs) || !isfinite(EMPTY_US) || EMPTY_US <= 0 ||
      !isfinite(SLOPE_PERCENT_PER_US) || SLOPE_PERCENT_PER_US <= 0 ||
      !isfinite(INTERCEPT_PERCENT) || !isfinite(MIN_DELTA_US) ||
      !isfinite(MAX_DELTA_US) || MIN_DELTA_US >= MAX_DELTA_US) return false;
  const double feature = medianUs-EMPTY_US;
  if (feature < MIN_DELTA_US || feature > MAX_DELTA_US) return false;
  moisture = SLOPE_PERCENT_PER_US*feature+INTERCEPT_PERCENT;
  return isfinite(moisture) && moisture >= 0 && moisture <= 100;
}
}
