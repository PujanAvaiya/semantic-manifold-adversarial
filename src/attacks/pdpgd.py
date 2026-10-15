"""PDPGD integration point.

The current archived notebook develops latent/tangent geometry and boundary-aware
warm starts, but it does not yet contain a cleaned reusable PDPGD implementation.
This module is intentionally left as an integration point rather than inventing an
implementation that was not part of the recorded baseline.
"""


def run_pdpgd(*args, **kwargs):
    raise NotImplementedError(
        "PDPGD has not yet been migrated into the modular baseline. "
        "Implement it only after the mathematical objective and experiment "
        "protocol are fixed in docs/attack_method.md."
    )
