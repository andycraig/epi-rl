import main
import numpy as np
from numpy.testing import assert_equal

# Check same results for cartpole.
print(main.getActionOld(np.array([1.])))
assert_equal(main.getActionOld(np.array([1.])), main.getActionNew(np.array([1.]), 'cartpole')), "Cartpole"
assert_equal(main.getActionOld(np.array([0.])), main.getActionNew(np.array([0.]), 'cartpole')), "Cartpole"
# Check for epidemic.
assert_equal((0, np.array([1., 0])), main.getActionNew(np.array([1., 0.]), 'epidemic')), "Action 0"
assert_equal((1, np.array([0., 1.])), main.getActionNew(np.array([0., 1.]), 'epidemic')), "Action 1"
assert_equal((2, np.array([0., 0.])), main.getActionNew(np.array([0., 0.]), 'epidemic')), "Action 2"
