# Serial link to the ESP32 (firmware/cebra_leds/cebra_leds.ino).
#
# Packet, 9 bytes:  0xAA  seq  r1 g1 b1  r2 g2 b2  checksum
#   checksum = XOR of the 7 bytes from seq to b2
# With ack on, the ESP32 sends seq back after it sets the LEDs, so we can time the round trip.
import time

HEADER = 0xAA


class LedLink:
    def __init__(self, port, baud=115200, ack=False):
        import serial
        # ESP32 boards wire DTR/RTS to the reset (EN) and boot (GPIO0) pins. Opening the port
        # with pyserial's defaults can hold the board in reset, so set both low BEFORE opening.
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, baud, 0.1
        self.ser.dtr = self.ser.rts = False
        self.ser.open()
        self.ack = ack
        self.seq = 0
        self.ack_errors = 0
        time.sleep(2.5)  # opening can still reboot some boards; wait out boot + the LED self-test
        self.ser.reset_input_buffer()

    def send(self, rgb1, rgb2):
        """Sends both colors. Returns round-trip ms if ack is on, else None."""
        self.seq = (self.seq + 1) & 0xFF
        payload = bytes([self.seq, *rgb1, *rgb2])
        chk = 0
        for b in payload:
            chk ^= b
        t0 = time.perf_counter()
        self.ser.write(bytes([HEADER]) + payload + bytes([chk]))
        if not self.ack:
            return None
        reply = self.ser.read(1)
        if len(reply) != 1 or reply[0] != self.seq:
            self.ack_errors += 1
            self.ser.reset_input_buffer()
            return None
        return (time.perf_counter() - t0) * 1000

    def off(self):
        self.send((0, 0, 0), (0, 0, 0))

    def close(self):
        try:
            self.off()
        finally:
            self.ser.close()
