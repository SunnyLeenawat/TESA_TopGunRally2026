#pragma once
#include <Arduino.h>
#include <math.h>
#include "driver/gpio.h"

// Canonical source: copy identically to both sketches with sync_headers.py.
namespace CoffeeSensor {
constexpr int CHARGE = 25, COMPARATOR = 27, REFERENCE = 34, BUTTON = 26, LED = 2;
constexpr unsigned COUNT = 64;
constexpr uint32_t DISCHARGE_US = 2000, TIMEOUT_US = 5000;
constexpr uint32_t REF_MIN_MV = 850, REF_MAX_MV = 950;
volatile bool armed = false, crossed = false;
volatile uint32_t crossingCycles = 0;
bool clockReady = false;

struct Reading {
  double medianUs = NAN;
  double madUs = NAN;
  uint32_t referenceMv = 0;
  unsigned valid = 0;
  const char *status = "not_measured";
};

void ARDUINO_ISR_ATTR onCrossing() {
  if (armed && !crossed) {
    crossingCycles = ESP.getCycleCount();
    crossed = true;
  }
}

inline void begin() {
  pinMode(CHARGE, OUTPUT);
  digitalWrite(CHARGE, LOW);
  pinMode(COMPARATOR, INPUT); // External 2 kOhm to 3.3 V.
  pinMode(BUTTON, INPUT_PULLUP);
  pinMode(LED, OUTPUT);
  digitalWrite(LED, LOW);
  analogReadResolution(12);
  analogSetPinAttenuation(REFERENCE, ADC_6db);
  clockReady = setCpuFrequencyMhz(240) && getCpuFrequencyMhz() == 240;
  attachInterrupt(digitalPinToInterrupt(COMPARATOR), onCrossing, FALLING);
}

inline void sort(double *values) {
  for (unsigned i = 1; i < COUNT; ++i) {
    const double value = values[i];
    unsigned j = i;
    while (j && values[j-1] > value) { values[j] = values[j-1]; --j; }
    values[j] = value;
  }
}

inline Reading measure() {
  Reading result;
  gpio_set_level((gpio_num_t)CHARGE, 0);
  if (!clockReady || getCpuFrequencyMhz() != 240) { result.status = "clock_error"; return result; }
  result.referenceMv = analogReadMilliVolts(REFERENCE);
  if (result.referenceMv < REF_MIN_MV || result.referenceMv > REF_MAX_MV) {
    result.status = "reference_error";
    return result;
  }
  double times[COUNT];
  for (unsigned i = 0; i < COUNT; ++i) {
    armed = false;
    gpio_set_level((gpio_num_t)CHARGE, 0);
    delayMicroseconds(DISCHARGE_US);
    if (digitalRead(COMPARATOR) != HIGH) { result.status = "initial_low"; return result; }
    crossed = false;
    armed = true;
    const uint32_t start = ESP.getCycleCount();
    gpio_set_level((gpio_num_t)CHARGE, 1);
    while (!crossed && uint32_t(ESP.getCycleCount()-start) < 240UL*TIMEOUT_US) {}
    armed = false;
    const bool captured = crossed;
    const uint32_t elapsed = crossingCycles-start;
    gpio_set_level((gpio_num_t)CHARGE, 0);
    if (!captured || elapsed == 0 || elapsed >= 240UL*TIMEOUT_US) {
      result.status = "timeout_or_spurious_edge";
      return result;
    }
    times[i] = double(elapsed)/240.0;
    ++result.valid;
  }
  sort(times);
  result.medianUs = (times[31]+times[32])/2.0;
  for (unsigned i = 0; i < COUNT; ++i) times[i] = fabs(times[i]-result.medianUs);
  sort(times);
  result.madUs = (times[31]+times[32])/2.0;
  result.status = "ok";
  return result;
}

// A held button never repeats. Boot requires a debounced release first.
inline bool buttonPressed() {
  static int previous = HIGH, stable = LOW;
  static bool enabled = false;
  static uint32_t changedAt = millis();
  const int now = digitalRead(BUTTON);
  if (now != previous) { previous = now; changedAt = millis(); }
  if (uint32_t(millis()-changedAt) >= 30 && now != stable) {
    stable = now;
    if (stable == HIGH) enabled = true;
    else if (enabled) { enabled = false; return true; }
  }
  return false;
}
}
