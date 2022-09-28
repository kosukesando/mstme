import sys


def is_interactive():
    ip = False
    if "ipykernel" in sys.modules:
        ip = True
    elif "IPython" in sys.modules:
        ip = True
    return ip
