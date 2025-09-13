# Helper functions to read and preprocess data files from Matlab format

# Data science libraries
import scipy
import scipy.io
import numpy as np
import pandas as pd

# Others
from pathlib import Path
from tqdm.auto import tqdm
import requests
import re

# ------------------------------------------------------------------------
# File loading (DE-only)
# ------------------------------------------------------------------------
def matfile_to_dic(folder_path):
    
    folder_path = Path(folder_path)
    output_dic = {}
    found_any = False
    for filepath in folder_path.rglob('*.mat'):  # recursive search
        found_any = True
        key_name = filepath.name
        output_dic[key_name] = scipy.io.loadmat(filepath)
    if not found_any:
        raise FileNotFoundError(
            f"No .mat files found under: {folder_path.resolve()}."
        )
    return output_dic


def remove_dic_items(dic):
    
    for _, values in dic.items():
        for k in ['__header__', '__version__', '__globals__']:
            if k in values:
                del values[k]


def _find_de_key(d):
    
    for k in d.keys():
        if 'DE_time' in k:
            return k
    return None


def label(filename):
    """
    Infer coarse label from filename.
    """
    if filename.startswith('B'):
        return 'B'
    elif 'IR' in filename:
        return 'IR'
    elif 'OR' in filename:
        return 'OR'
    elif 'Normal' in filename:
        return 'N'


def fault_size_from_name(fname):
    
    m = re.search(r'(?:B|IR|OR)[_\-\s@]?0*(007|014|021)', fname, flags=re.IGNORECASE)
    return m.group(1) if m else None


# ------------------------------------------------------------------------
# DataFrame preparation (DE-only)
# ------------------------------------------------------------------------
def matfile_to_df(folder_path):
    
    dic = matfile_to_dic(folder_path)
    remove_dic_items(dic)

    slim = {}
    skipped = []
    for fname, d in dic.items():
        de_key = _find_de_key(d)
        if de_key is None:
            skipped.append(fname)
            continue
        slim[fname] = {'DE_time': d[de_key].squeeze()}

    if not slim:
        raise RuntimeError(
            "No DE_time keys found in the provided directory. "
            "Ensure your folder contains 12 kHz DE recordings."
        )

    df = pd.DataFrame.from_dict(slim, orient='index').reset_index()
    df = df.rename(columns={'index': 'filename'})
    df['label'] = df['filename'].apply(label)
    return df


def divide_signal(df, segment_length, overlap=0, per_window_norm=False):
    
    hop = segment_length - overlap
    assert hop > 0, "overlap must be < segment_length"

    dic = {}
    idx = 0
    for i in range(df.shape[0]):
        signal = np.asarray(df.iloc[i]['DE_time']).reshape(-1)
        n = len(signal)
        for start in range(0, n - segment_length + 1, hop):
            end = start + segment_length
            w = signal[start:end]
            if per_window_norm:
                mu = w.mean()
                sig = w.std()
                w = (w - mu) / (sig + 1e-8)
            dic[idx] = {
                "signal": w.astype(np.float32),
                "label": df.iloc[i]['label'],
                "filename": df.iloc[i]['filename'],
            }
            idx += 1

    if not dic:
        raise RuntimeError("divide_signal produced zero windows. Check segment_length/overlap.")

    df_tmp = pd.DataFrame.from_dict(dic, orient="index")
    df_output = pd.concat(
        [df_tmp[["label", "filename"]],
         pd.DataFrame(np.vstack(df_tmp["signal"].values))],
        axis=1
    )
    return df_output


def normalize_signal(df):
    
    def _z(x: np.ndarray):
        x = np.asarray(x).reshape(-1).astype(np.float32)
        s = x.std()
        return (x - x.mean()) / (s + 1e-8)
    df['DE_time'] = df['DE_time'].apply(_z)


def get_df_all(data_path, segment_length=2048, normalize=False, overlap=0, per_window_norm=False):
    
    df_files = matfile_to_df(data_path)
    if normalize:
        normalize_signal(df_files)

    df_processed = divide_signal(
        df_files,
        segment_length,
        overlap=overlap,
        per_window_norm=per_window_norm
    )

    map_label = {'N': 0, 'B': 1, 'IR': 2, 'OR': 3}
    df_processed['label'] = df_processed['label'].map(map_label)
    return df_processed


# ------------------------------------------------------------------------
# Optional: downloader
# ------------------------------------------------------------------------
def download(url: str, dest_dir: Path, save_name: str, suffix=None) -> Path:
   
    assert isinstance(dest_dir, Path), "dest_dir must be a Path object"
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split('/')[-1] if save_name is None else (save_name + (suffix or ''))
    file_path = dest_dir / filename
    if not file_path.exists():
        print(f"Downloading {file_path}")
        with open(f'{file_path}', 'wb') as f:
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length', 0))
            with tqdm(total=total, unit='B', unit_scale=True, desc=filename) as pbar:
                for data in response.iter_content(chunk_size=1024 * 1024):
                    f.write(data)
                    pbar.update(len(data))
    return file_path


# ------------------------------------------------------------------------
# Torch Dataset (supports overlap, size filtering, grouped split)
# ------------------------------------------------------------------------
import torch
from torch.utils import data as torch_data
from torch.utils.data import Dataset

class CWRUDataset(Dataset):
    
    def __init__(
        self,
        data_path,
        segment_length=2048,
        normalize=True,
        load_id=0,
        overlap=0,
        size_filter=None,
        per_window_norm=False
    ):
        df_all = get_df_all(
            Path(data_path),
            segment_length=segment_length,
            normalize=normalize,
            overlap=overlap,
            per_window_norm=per_window_norm
        )

        # Filter by load id (expects filenames like *_<load_id>.mat)
        df_all = df_all[df_all["filename"].str.contains(fr"_{load_id}\.mat$", regex=True)]

        # Optional: filter by fault size (e.g., '007', '014', '021')
        if size_filter is not None:
            df_all = df_all[df_all["filename"].apply(fault_size_from_name) == str(size_filter)]

        df_all = df_all.reset_index(drop=True)

        # Store features/labels
        self.X = df_all.iloc[:, 2:].values.astype(np.float32)
        self.y = df_all["label"].values.astype(np.int64)
        self.n_in = segment_length

        # Keep filenames for grouped splitting (avoids leakage by file)
        self.filenames = df_all["filename"].values.astype(str)

        if len(self.X) == 0:
            raise ValueError(
                f"No samples found for load_id={load_id} with size_filter={size_filter}. "
                f"Check filenames and data_path."
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y