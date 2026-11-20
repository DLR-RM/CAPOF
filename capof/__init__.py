"""CAPOF — Context-Adaptive Policy Framework.

CAPOF is a context-adaptive framework that leverages the epistemic uncertainty
captured from demonstrations to enable robust and reactive robotic manipulation.
The framework combines a  Gaussian-process LfD policy with a stabilizing policy
(epistemic-uncertainty driven) and a goal-attractor policy (kernel driven) into
a mixture of experts that adapts to the current context.

Public API:
    - :class:`~capof.capof.ContextAdaptivePolicyFramework`: the policy framework.
    - :class:`~capof.gp_scikit.GP`: Gaussian process regressor with epistemic
      uncertainty and gradient support.
    - :class:`~capof.outer_control_loop.OuterControlLoop`: base class for the
      execution loop (robot / simulation interfaces).
    - :func:`~capof.training.train_capof`: shared GP fitting helper.
"""

from capof.capof import ContextAdaptivePolicyFramework
from capof.gp_scikit import GP
from capof.outer_control_loop import OuterControlLoop
from capof.training import train_capof

__all__ = ["ContextAdaptivePolicyFramework", "GP", "OuterControlLoop", "train_capof"]
