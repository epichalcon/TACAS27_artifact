import pathlib
import re

def get_last_version(path: str, name:str):
    dirs    = pathlib.Path(path).glob(f"{name}[0-9]*")
    highest = max(
        (int(re.search(r"(\d+)", d.name).group(1)) for d in dirs),
        default=0,
    )
    return highest