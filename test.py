# %%
import sys
import os
from pathlib import Path
import re
import shutil
# %%
for fp0 in Path("ww3_meteo").glob("*.nc"):
    _id0 = fp0.name.replace(fp0.suffix, "").replace("_ugrid_wave_meteo", "")
    for fp1 in Path("track").glob("**/*.txt"):
        _id1 = fp1.name.replace(fp1.suffix, "")
        _cat = int(re.search("(?<=Cat)[0-9]", _id1).group(0))
        _id1 = re.sub(r"Cyclone_Cat..", "", _id1)
        if _id0 == _id1:
            print(_cat)
            # with
            break
# %%
