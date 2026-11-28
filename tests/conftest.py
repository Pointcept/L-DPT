import os

# CPU only: the tests never need a GPU, and several compare against CPU results, so a
# machine with a visible GPU (CUDA_VISIBLE_DEVICES set) still runs them on the CPU.
# LDPT_TEST_CUDA=1 keeps the GPUs visible for the CUDA variants of the option tests
# (test_l_dpt_options.py, test_structural_options.py); run those files alone then.
if os.environ.get("LDPT_TEST_CUDA") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
