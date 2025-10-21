import sys, os

print("Python:", sys.version)
print("Executable:", sys.executable)
print("CWD:", os.getcwd())
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("PATH snippet:", os.environ.get("PATH", "")[:200], "...")

try:
    import numpy, pandas, scipy, cvxpy
    print("NumPy:", numpy.__version__)
    print("Pandas:", pandas.__version__)
    print("SciPy:", scipy.__version__)
    print("CVXPY:", cvxpy.__version__)
    print("OK: all libs import.")
except Exception as e:
    print("IMPORT ERROR:", repr(e))