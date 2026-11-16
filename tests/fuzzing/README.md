# On-demand Mellinger comparison

Compare the Crazyflow Mellinger controller and motor mixer with a pinned
Crazyflie firmware revision. The test checks signed motor commands before
clipping and battery compensation.

## Run the test

```sh
pixi run fuzzing
```

The fuzzing environment builds the controller and mixer of the pinned
[Bitcraze firmware](https://github.com/bitcraze/crazyflie-firmware) commit into a Python module.

## What it checks

The test runs 100 inputs drawn with a fixed seed and a 20-step sequence.
It allows up to **3 PWM units** when both mixers get identical inputs and
up to **8 PWM units** when they get their respective controller outputs
in comparable steps, after the controller checks pass.

For the matched comparison, the test sets the vertical D gain to the
firmware value of 0.4. It reports the Crazyflow default of 0.5 separately.
The sequence also records how each controller handles its stored integral
state when thrust is zero or negative. With the observed reset difference,
controller outputs are compared through the reset step. The identical-input
mixer and replay checks cover all 20 steps.

The output includes the maximum errors, any stock gain or reset differences,
and a JSON result prefixed with `MELLINGER_FUZZ_RESULT=`. The scope is the
controller and raw mixer with zero setpoint body rates and a short sequence.
