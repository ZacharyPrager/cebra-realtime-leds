// Two RGB LEDs driven by the laptop over USB serial.
//   LED 1 = where the rat really is
//   LED 2 = where CEBRA thinks it is
//
// Packet from the laptop (9 bytes):  0xAA  seq  r1 g1 b1  r2 g2 b2  checksum
//   checksum = XOR of the 7 bytes from seq to b2
// After a good packet we send seq back, so the laptop can time the round trip.
//
// Builds with PlatformIO (firmware/platformio.ini) or the Arduino IDE,
// on Arduino-ESP32 core 2.x or 3.x (their PWM functions differ, see pwmInit/pwmWrite).

#include <Arduino.h>
#include <esp_arduino_version.h>
#include <math.h>

const bool COMMON_ANODE = true;   // true if your LEDs' shared leg goes to 3V3 instead of GND

// LED 1: R, G, B    LED 2: R, G, B
const int PINS[6] = {25, 26, 27, 32, 33, 13};

const uint32_t BAUD = 115200;
const uint32_t PWM_FREQ = 5000;
const uint8_t PWM_BITS = 8;
const uint32_t TIMEOUT_MS = 1000;  // no data for this long -> LEDs off
const uint8_t HEADER = 0xAA;

uint8_t gammaTable[256];  // makes brightness steps look even to your eye
uint8_t packet[8];        // seq, r1, g1, b1, r2, g2, b2, checksum
int filled = -1;          // -1 = waiting for a header byte
uint32_t lastPacketMs = 0;

void pwmInit(int i) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PINS[i], PWM_FREQ, PWM_BITS);  // 3.x: PWM is set up by pin
#else
  ledcSetup(i, PWM_FREQ, PWM_BITS);         // 2.x: PWM channel i, then attach the pin
  ledcAttachPin(PINS[i], i);
#endif
}

void pwmWrite(int i, uint8_t duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PINS[i], duty);
#else
  ledcWrite(i, duty);
#endif
}

void setLed(int led, uint8_t r, uint8_t g, uint8_t b) {
  uint8_t c[3] = {r, g, b};
  for (int i = 0; i < 3; i++) {
    uint8_t duty = gammaTable[c[i]];
    if (COMMON_ANODE) duty = 255 - duty;
    pwmWrite(led * 3 + i, duty);
  }
}

void allOff() {
  setLed(0, 0, 0, 0);
  setLed(1, 0, 0, 0);
}

void selfTest() {  // red, green, blue on each LED so you can check the wiring
  for (int led = 0; led < 2; led++) {
    setLed(led, 255, 0, 0); delay(200);
    setLed(led, 0, 255, 0); delay(200);
    setLed(led, 0, 0, 255); delay(200);
    setLed(led, 0, 0, 0);
  }
}

void setup() {
  Serial.begin(BAUD);
  for (int i = 0; i < 256; i++) gammaTable[i] = (uint8_t)(pow(i / 255.0, 2.2) * 255 + 0.5);
  for (int i = 0; i < 6; i++) pwmInit(i);
  allOff();
  selfTest();
}

void loop() {
  while (Serial.available()) {
    uint8_t b = Serial.read();
    if (filled < 0) {  // looking for the start of a packet
      if (b == HEADER) filled = 0;
      continue;
    }
    packet[filled++] = b;
    if (filled < 8) continue;

    filled = -1;
    uint8_t chk = 0;
    for (int i = 0; i < 7; i++) chk ^= packet[i];
    if (chk != packet[7]) continue;  // corrupted: drop it and wait for the next header

    setLed(0, packet[1], packet[2], packet[3]);
    setLed(1, packet[4], packet[5], packet[6]);
    Serial.write(packet[0]);  // ack with seq
    lastPacketMs = millis();
  }

  if (lastPacketMs && millis() - lastPacketMs > TIMEOUT_MS) {
    allOff();
    lastPacketMs = 0;
  }
}
