
import platform
import sys

import matplotlib
import numpy
import pandas
import scipy
import sklearn



def main():
    print("Lab 2 environment information")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {numpy.__version__}")
    print(f"Pandas: {pandas.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")



if __name__ == "__main__":
    main()
