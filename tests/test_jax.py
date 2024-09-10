import jax
import os

jax.config.update('jax_enable_x64', True)

print(jax.devices())