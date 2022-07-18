import os

for thr in [1.0, 1.5, 2.0]:
    os.system(f"python mstme.py {thr} 15 45 -r guadeloupe")
