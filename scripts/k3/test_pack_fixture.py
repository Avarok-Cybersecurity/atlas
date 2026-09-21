# SPDX-License-Identifier: AGPL-3.0-only
import unittest
try:
    import numpy as np
except ImportError:
    np = None
from pack_fixture import pack

@unittest.skipIf(np is None, 'fixture quantization requires numpy')
class PackTests(unittest.TestCase):
    def test_exact_representable_levels_and_scale(self):
        values=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]*2,dtype=np.float32)[None,:]
        packed,scales=pack(values*4)
        self.assertEqual(scales.tolist(),[[129]])
        codes=np.empty_like(values,dtype=np.uint8)
        codes[:,::2]=packed&15;codes[:,1::2]=packed>>4
        levels=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32)
        decoded=levels[codes&7]*np.where(codes&8,-1,1)*4
        np.testing.assert_array_equal(decoded,values*4)
        self.assertEqual(pack(np.zeros((1,32),dtype=np.float32))[1].tolist(),[[127]])

    def test_invalid_shape_and_nonfinite_are_refused(self):
        for weight in (np.zeros((1,31)),np.full((1,32),np.nan)):
            with self.assertRaises(ValueError):pack(weight)
