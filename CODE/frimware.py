import time
import board
import digitalio
import usb_hid

from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keycode import Keycode

# Change these pins to whatever your 3 buttons are wired to
BUTTON_PINS = [board.GP0, board.GP1, board.GP2]
KEYCODES = [Keycode.P, Keycode.Q, Keycode.X]

keyboard = Keyboard(usb_hid.devices)

buttons = []
for pin in BUTTON_PINS:
    button = digitalio.DigitalInOut(pin)
    button.direction = digitalio.Direction.INPUT
    button.pull = digitalio.Pull.UP
    buttons.append(button)

last_states = [True, True, True]  # True = not pressed (because pull-up)

while True:
    for i, button in enumerate(buttons):
        current_state = button.value  # True when released, False when pressed

        # Detect a new press
        if last_states[i] and not current_state:
            keyboard.send(KEYCODES[i])

        last_states[i] = current_state

    time.sleep(0.01)
