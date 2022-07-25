import unittest

from rng.mrg32k3a import *

A1p127 = [[2427906178, 3580155704, 949770784],
          [226153695, 1230515664, 3580155704],
          [1988835001,  986791581, 1230515664]
          ]

A2p127 = [[1464411153,  277697599, 1610723613],
          [32183930, 1464411153.0, 1022607788],
          [2824425944, 32183930.0, 2093834863]
          ]

A1p76 = [[82758667, 1871391091, 4127413238],
         [3672831523, 69195019, 1871391091],
         [3672091415, 3528743235, 69195019]
         ]

A2p76 = [[1511326704, 3759209742, 1610795712],
         [4292754251, 1511326704, 3889917532],
         [3859662829, 4292754251, 3708466080],
         ]

seed = (12345, 12345, 12345, 12345, 12345, 12345)


class TestMRG32k3a(unittest.TestCase):

    def test_A1p127(self):
        self.assertEqual(mat33_power_mod(A1p0, 2**127, mrgm1), A1p127)

    def test_A2p127(self):
        self.assertEqual(mat33_power_mod(A2p0, 2**127, mrgm2), A2p127)

    def test_A1p76(self):
        self.assertEqual(mat33_power_mod(A1p0, 2**76, mrgm1), A1p76)

    def test_A2p76(self):
        self.assertEqual(mat33_power_mod(A2p0, 2**76, mrgm2), A2p76)

    def test_A1p47(self):
        self.assertEqual(mat33_power_mod(A1p0, 2**47, mrgm1), A1p47)

    def test_A2p47(self):
        self.assertEqual(mat33_power_mod(A2p0, 2**47, mrgm2), A2p47)

    def test_A1p94(self):
        self.assertEqual(mat33_power_mod(A1p0, 2**94, mrgm1), A1p94)

    def test_A2p94(self):
        self.assertEqual(mat33_power_mod(A2p0, 2**94, mrgm2), A2p94)

    def test_A1p141(self):
        self.assertEqual(mat33_power_mod(A1p0, 2**141, mrgm1), A1p141)

    def test_A2p141(self):
        self.assertEqual(mat33_power_mod(A2p0, 2**141, mrgm2), A2p141)

    def test_get_current_state(self):
        rng = MRG32k3a()
        self.assertEqual(rng.get_current_state(), seed)

    def test_first_state(self):
        rng = MRG32k3a()
        self.assertEqual(rng._current_state, seed)

    def test_second_state(self):
        rng = MRG32k3a()
        rng.random()
        st1 = mat31_mod(mat33_mat31_mult(A1p0, seed[0:3]), mrgm1)
        st2 = mat31_mod(mat33_mat31_mult(A2p0, seed[3:6]), mrgm2)
        self.assertSequenceEqual(rng._current_state, st1 + st2)

    def test_third_state(self):
        rng = MRG32k3a()
        rng.random()
        rng.random()
        A1sq = mat33_mat33_mult(A1p0, A1p0)
        A2sq = mat33_mat33_mult(A2p0, A2p0)
        st1 = mat31_mod(mat33_mat31_mult(A1sq, seed[0:3]), mrgm1)
        st2 = mat31_mod(mat33_mat31_mult(A2sq, seed[3:6]), mrgm2)
        self.assertSequenceEqual(rng._current_state, st1 + st2)

    def test_hundreth_state(self):
        rng = MRG32k3a()
        for _ in range(99):
            rng.random()
        st1 = mat31_mod(mat33_mat31_mult(mat33_power_mod(A1p0, 99, mrgm1), seed[0:3]), mrgm1)
        st2 = mat31_mod(mat33_mat31_mult(mat33_power_mod(A2p0, 99, mrgm2), seed[3:6]), mrgm2)
        self.assertSequenceEqual(rng._current_state, st1 + st2)

    def test_advance_stream(self):
        rng = MRG32k3a(s_ss_sss_index=[0, 1, 1])
        rng.advance_stream()
        rng2 = MRG32k3a(s_ss_sss_index=[1, 0, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng._current_state)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [1, 0, 0])

    def test_advance_substream(self):
        rng = MRG32k3a(s_ss_sss_index=[0, 0, 1])
        rng.advance_substream()
        rng2 = MRG32k3a(s_ss_sss_index=[0, 1, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, seed)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [0, 1, 0])

    def test_advance_subsubstream(self):
        rng = MRG32k3a()
        rng.advance_subsubstream()
        rng2 = MRG32k3a(s_ss_sss_index=[0, 0, 1])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, seed)
        self.assertEqual(rng.substream_start, seed)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [0, 0, 1])

    def test_reset_stream(self):
        rng = MRG32k3a(s_ss_sss_index=[1, 1, 1])
        rng.random()
        rng.reset_stream()
        rng2 = MRG32k3a(s_ss_sss_index=[1, 0, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng._current_state)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [1, 0, 0])

    def test_reset_substream(self):
        rng = MRG32k3a(s_ss_sss_index=[1, 1, 1])
        rng.random()
        rng.reset_substream()
        rng2 = MRG32k3a(s_ss_sss_index=[1, 1, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        rng3 = MRG32k3a(s_ss_sss_index=[1, 0, 0])
        self.assertEqual(rng.stream_start, rng3._current_state)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [1, 1, 0])

    def test_reset_subsubstream(self):
        rng = MRG32k3a(s_ss_sss_index=[1, 1, 1])
        rng.random()
        rng.reset_subsubstream()
        rng2 = MRG32k3a(s_ss_sss_index=[1, 1, 1])
        self.assertEqual(rng._current_state, rng2._current_state)
        rng3 = MRG32k3a(s_ss_sss_index=[1, 0, 0])
        rng4 = MRG32k3a(s_ss_sss_index=[1, 1, 0])
        self.assertEqual(rng.stream_start, rng3._current_state)
        self.assertEqual(rng.substream_start, rng4._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [1, 1, 1])

    def test_init_fixed_s_ss_sss(self):
        rng = MRG32k3a(s_ss_sss_index=[1, 1, 1])
        rng2 = MRG32k3a()
        rng2.start_fixed_s_ss_sss([1, 1, 1])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng2.stream_start)
        self.assertEqual(rng.substream_start, rng2.substream_start)
        self.assertEqual(rng.subsubstream_start, rng2.subsubstream_start)
        self.assertEqual(rng.s_ss_sss_index, rng2.s_ss_sss_index)

    def test_jump_fixed_s_ss_sss(self):
        rng = MRG32k3a()
        rng.start_fixed_s_ss_sss([1, 1, 1])
        rng2 = MRG32k3a()
        rng2.advance_stream()
        rng2.advance_substream()
        rng2.advance_subsubstream()
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng2.stream_start)
        self.assertEqual(rng.substream_start, rng2.substream_start)
        self.assertEqual(rng.subsubstream_start, rng2.subsubstream_start)
        self.assertEqual(rng.s_ss_sss_index, rng2.s_ss_sss_index)

if __name__ == '__main__':
    unittest.main()