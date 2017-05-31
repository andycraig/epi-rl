import main
import numpy as np
from numpy.testing import assert_equal

print(main.getActionNew(np.array([1.])))
assert_equal(main.getActionOld(np.array([1.])), main.getActionNew(np.array([1.])))
assert_equal(main.getActionOld(np.array([0.])), main.getActionNew(np.array([0.])))
print("want 1, [0,0]")
print(main.getActionNew(np.array([1., 0.])))
print("want 2, [1,0]")
print(main.getActionNew(np.array([0, 1.])))
