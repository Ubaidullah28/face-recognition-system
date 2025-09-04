import os
import re
from typing import Tuple


ROOT_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')


SAFE_CHARS = re.compile(r"[^A-Za-z0-9_. -]")




def safe_person_name(name: str) -> str:
    name = name.strip()
    name = SAFE_CHARS.sub("_", name)
    # Collapse spaces
    name = re.sub(r"\s+", " ", name)
    return name[:64] if name else "person"




def person_dir(name: str) -> str:
    return os.path.join(DATASET_DIR, safe_person_name(name))




def ensure_person_dir(name: str) -> str:
    pdir = person_dir(name)
    os.makedirs(pdir, exist_ok=True)
    return pdir