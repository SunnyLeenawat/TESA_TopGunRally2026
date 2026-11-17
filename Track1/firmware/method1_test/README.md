# Method 1 bench test

Use a DOIT ESP32 DEVKIT V1 (ESP32-WROOM-32) and the supplied LM393. This experiment records raw RC crossing times. It is not the final competition firmware and does not claim calibrated pF values or implement the competition LED/output format yet.

## Dependencies and Arduino IDE setup

`sketch.yaml` records the board/core dependency as an Arduino CLI build profile. For Arduino IDE, install and select the matching package manually using the steps below; do not rely on the YAML file to install it or set the IDE board automatically.

| Dependency | Selected version / setting |
| --- | --- |
| Arduino IDE | 2.x; record your exact installed version with experiment results |
| Boards Manager package | **esp32 by Espressif Systems**, version **3.3.0** |
| Board | **DOIT ESP32 DEVKIT V1** for the original ESP32-WROOM-32 development board |
| Board identifier (FQBN) | `esp32:esp32:esp32doit-devkit-v1` |
| Additional Library Manager libraries | **None** |
| Serial Monitor | **9600 baud** |

`Arduino.h`, `driver/gpio.h`, and the ESP32 APIs used by this sketch are provided by the ESP32 board package. Do not install a separate GPIO or LM393 library.

1. Open **File > Preferences** and add this to **Additional Boards Manager URLs**, preserving any existing entries:

   ```text
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```

2. Open **Boards Manager**, search for **esp32**, choose **esp32 by Espressif Systems**, select **3.3.0**, and install it.
3. Open `method1_test.ino`. Under **Tools > Board > esp32**, select **DOIT ESP32 DEVKIT V1**, then select the connected COM port.
4. Click **Verify**, then **Upload**. Record the board settings before gathering calibration data.

Version 3.3.0 is the selected reproducible baseline, not a claim of a successful build or the latest release. Compilation and hardware verification remain pending. If the team chooses another core version, update this table and `sketch.yaml` together and recheck timing calibration.

Optional Arduino CLI equivalent, from the repository root:

```powershell
arduino-cli compile --profile doit_esp32_devkit_v1 Track1/firmware/method1_test
```

References: [Espressif IDE installation](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html), [ESP32 core 3.3.0 release](https://github.com/espressif/arduino-esp32/releases/tag/3.3.0), and [Arduino build profiles](https://docs.arduino.cc/arduino-cli/sketch-project-file/).

## Wiring (power disconnected)

For an LM393 DIP-8, verify the notch/dot and pin numbering before connecting:

| Connection | Destination |
| --- | --- |
| LM393 pin 8 | ESP32 3.3 V |
| LM393 pin 4 | GND |
| 100 nF capacitor | Between LM393 pins 8 and 4, close to the chip |
| 10 kOhm potentiometer outer terminals | 3.3 V and GND |
| Potentiometer wiper | LM393 pin 3 (+) and ESP32 GPIO34 |
| ESP32 GPIO25 | 1 MOhm resistor, then sensing node |
| Sensing node | LM393 pin 2 (-) and one terminal of test capacitor |
| Other test capacitor terminal | GND |
| LM393 pin 1 (output) | GPIO27 and 10 kOhm pullup resistor to 3.3 V |
| Momentary button | GPIO26 to GND |
| Unused comparator | Pin 5 to GND, pin 6 to potentiometer wiper; leave pin 7 unconnected |

Use short, fixed wires. GPIO34 measures only the reference, not the capacitor node. The output falls when the charging voltage crosses the reference. Never pull the output up to 5 V.

The slide's 2.086 V reference is outside the ordinary LM393 common-mode range at a 3.3 V supply. Use about **900 mV** here, estimated by the ESP32 ADC. This is not a precision voltage measurement. See the [LM393 datasheet](https://www.ti.com/lit/ds/symlink/lm393.pdf).

## Run the first experiment

1. Open `method1_test.ino` in Arduino IDE. Select the actual ESP32 board and port, and record the installed ESP32 core version. This sketch needs only the ESP32 core.
2. Upload and open Serial Monitor at **9600 baud**.
3. Send `r`, adjust the potentiometer toward 900 mV, and repeat until stable. The sketch accepts 700-1100 mV.
4. With the test capacitor disconnected, send `m` once to record the open-fixture baseline. Avoid touching the sensing node.
5. Disconnect power, install the **5 pF** capacitor, and reconnect. Send `r` to verify the reference. Press and release the button once, or send `m`.
6. Wait for all 64 CSV rows to finish printing before the next press. Save the output with the capacitor label, session, and circuit configuration. A batch takes several seconds to print at 9600 baud; printing occurs after acquisition.
7. Collect five batches, then repeat for 10, 15, and 20 pF. Repeat across power-cycle sessions as specified in Planning.md.

Keep raw data. `initial_low` means the comparator did not return high after discharge; check wiring, reference and pullup. `timeout` means no crossing was captured in 1 ms. Invalid rows have blank elapsed time.

The recorded interval includes GPIO-start and interrupt latency, comparator delay, and parasitic capacitance. CPU-cycle timing improves resolution but does not eliminate those errors. The small reference capacitors have +/-5% tolerance. Do not use `C = t/R` at this threshold or label timing readings as measured pF.

First plot elapsed time against nominal capacitance with horizontal +/-5% reference bars and vertical repeatability bars. Also plot readings in sequence and count failures. Fit timing-to-capacitance calibration only after checking that the four capacitors produce distinguishable readings. Keep this firmware/configuration and wire arrangement fixed during calibration.

This sketch has not yet been compiled against the team's Arduino installation or tested on physical hardware.
