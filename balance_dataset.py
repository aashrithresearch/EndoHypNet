# -*- coding: utf-8 -*-

from pathlib import Path
import shutil, random

def balance_dataset(data_path, strategy='oversample'):
    target_dir = Path('/content/balanced_data')
    target_dir.mkdir(parents=True, exist_ok=True)

    classes = [cls.name for cls in data_path.ls() if cls.is_dir()]
    counts = {cls: len((data_path/cls).ls(file_exts=['.JPG', '.jpeg', '.jpg'])) for cls in classes}
    print("Original class counts:", counts)

    max_count = max(counts.values()) if strategy == 'oversample' else min(counts.values())

    for cls in classes:
        cls_path = data_path/cls
        dest_cls_path = target_dir/cls
        dest_cls_path.mkdir(parents=True, exist_ok=True)

        files = list(cls_path.ls(file_exts=['.JPG', '.jpeg', '.jpg']))
        n_needed = max_count

        sampled = random.choices(files, k=n_needed) if strategy == 'oversample' else random.sample(files, k=n_needed)

        for f in sampled:
            new_name = f"{f.stem}_{random.randint(10000,99999)}{f.suffix}"
            shutil.copy(f, dest_cls_path/new_name)

    print(f"Balanced dataset saved to: {target_dir}")
    return target_dir
