#include "coffee_sensor.h"
#include "moisture_calibration.h"
#include <string.h>

unsigned remaining = 0, runId = 0;
uint32_t nextAt = 0;
char command[16];
unsigned length = 0;
bool overflow = false;

void report() {
  const CoffeeSensor::Reading r = CoffeeSensor::measure();
  Serial.printf("COFFEE_DATA,%u,", ++runId);
  if (isfinite(r.medianUs)) Serial.print(r.medianUs, 6);
  Serial.print(',');
  if (isfinite(r.madUs)) Serial.print(r.madUs, 6);
  Serial.printf(",%lu,%u,%s\n", (unsigned long)r.referenceMv, r.valid, r.status);
  double moisture;
  Serial.print("Moisture estimate = ");
  if (r.valid == CoffeeSensor::COUNT && MoistureCalibration::predict(r.medianUs, moisture)) {
    Serial.print(moisture, 2);
    Serial.println(" %");
  } else Serial.println("UNAVAILABLE (uncalibrated, invalid, or out of range)");
}

void setup() {
  Serial.begin(9600);
  CoffeeSensor::begin();
  Serial.println("Coffee test: run + Enter = 15 batches; m = one; r = reference; stop = cancel");
  Serial.println("COFFEE_HEADER,run,median_us,mad_us,reference_mv,valid,status");
}

void loop() {
  if (CoffeeSensor::buttonPressed() && remaining == 0) report();
  while (Serial.available()) {
    const char ch = Serial.read();
    if (ch == '\r' || ch == '\n') {
      command[length] = '\0';
      if (overflow) Serial.println("COMMAND_TOO_LONG");
      else if (!strcmp(command, "stop")) { remaining = 0; Serial.println("RUN_STOPPED"); }
      else if (remaining && length) Serial.println("BUSY");
      else if (!strcmp(command, "run")) {
        runId = 0; remaining = 15; nextAt = millis(); Serial.println("RUN_START,15");
      } else if (!strcmp(command, "m")) report();
      else if (!strcmp(command, "r")) Serial.println(analogReadMilliVolts(CoffeeSensor::REFERENCE));
      else if (length) Serial.println("USE run, m, r, stop THEN ENTER");
      length = 0; overflow = false;
    } else if (ch == '\b' || ch == 127) { if (length) --length; }
    else if (ch >= 32 && ch <= 126) {
      if (length < sizeof(command)-1) command[length++] = ch;
      else overflow = true;
    }
  }
  if (remaining && int32_t(millis()-nextAt) >= 0) {
    report(); --remaining; nextAt = millis()+500;
    if (!remaining) Serial.println("RUN_DONE,15");
  }
}
