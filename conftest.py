collect_ignore = ["tests/test_baseline.py"]

import warnings


def pytest_configure(config):
    # Suppress third-party RuntimeWarnings triggered by deliberate edge-case inputs
    # (e.g. wilcoxon on all-zero differences, np.mean on empty slice).
    # These do not indicate bugs — our code handles all such cases explicitly.
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\..*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\..*")
