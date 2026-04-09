import os

from gst.itest import test_gst_itest
from gcoreg2.itest import test_gstcoreg2_itest
from par1.itest import test_par1_itest
from pr.itest import test_pr_itest

# run_test_scripts = {
#     "gst/itest.py": ["seq", "par_f", "par_s"],
#     "gcoreg/itest.py": ["seq", "par_f", "par_s"],
#     "par1/itest.py": ["seq", "par_f"],
#     "pr/itest.py": ["seq"],
# }

itest_calls = {
    test_gst_itest: ["seq"],
    test_gstcoreg2_itest: ["seq"],
    test_par1_itest: ["seq"],
    test_pr_itest: ["seq"],
}

os.environ["ARRAY_MODULE"] = "cupy"  # "numpy" or "cupy"

if __name__ == "__main__":

    for itest, modes in itest_calls.items():
        print(f"{itest.__name__} in mode `{modes[0]}` returned: {itest()}")

        # for mode in modes:
        #     print(f"Running {itest.__name__} in {mode} mode...")
        #     command = f"ARRAY_MODULE={mode} python tests/integration/{script}"
        #     os.system(command)