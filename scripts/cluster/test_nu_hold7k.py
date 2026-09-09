"""Dependency-free tests for exact optimizer-update schedule boundaries."""
import math
from nu_hold7k_launch import nu_at_step

expected = {0: 1024, 6999: 1024, 7000: 512, 7999: 512,
            8000: 256, 9000: 128, 10000: 64, 11000: 32,
            11999: 32, 12000: 14, 17999: 14, 18000: 14}
for completed, value in expected.items():
    assert nu_at_step(completed) == value, (completed, value)
assert math.ceil(0.05555555555555555 * 18000) == 1000
assert sum(nu_at_step(i) == 1024 for i in range(18000)) == 7000
assert sum(nu_at_step(i) == 14 for i in range(18000)) == 6000
print('PASS: schedule boundaries, 7k warm hold, 6k terminal hold, 1k LR warmup')
