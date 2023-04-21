# coding: utf-8

# ====================================================
# imports
from pathlib import Path


# ====================================================
# code
def get_next_available_filename(filename: Path):
    if filename.is_dir():
        raise ValueError("filename is a directory.")

    dup_index = 0

    for existing_file in sorted(filename.parent.iterdir()):
        if existing_file.stem.startswith(filename.stem):
            if existing_file.stem[-1] == ')':
                dup_index = max(dup_index, int(existing_file.stem[:-1].split('(')[-1]) + 1)

            else:
                dup_index = 1

    if dup_index:
        return Path(filename.parent / (filename.stem + f"({dup_index}){filename.suffix}"))

    return filename
