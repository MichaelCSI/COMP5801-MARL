# Vendored subset of marlbenchmark/on-policy (see ../VENDORED.md).
#
# This is the ONLY file that differs from upstream. Upstream's
# onpolicy/__init__.py imports every environment (StarCraft II via absl,
# Hanabi, MPE, Football) and the wandb/tensorboardX runners, none of which
# this project vendors or needs. The MAPPO modules import each other by
# absolute path (onpolicy.algorithms..., onpolicy.utils...), so an otherwise
# empty package file is enough.
__version__ = "0.1.0"
