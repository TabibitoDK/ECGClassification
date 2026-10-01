import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
mpl.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_dir():
    data_dir = os.environ.get('ECG_DATA_DIR')
    if data_dir:
        return Path(data_dir).expanduser()
    return PROJECT_ROOT / 'data'


class Digitizer:
    def __init__(self, df_type, sr = 125):
        self.sr = sr
        self.df_type = df_type
        self.data_dir = _resolve_data_dir()
        csv_path = self.data_dir / f'mitbih_{df_type}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f'Could not find {csv_path}. Set ECG_DATA_DIR or use --data-dir to point to your dataset directory.')
        self.ecg_df = pd.read_csv(csv_path)
        self.df_len = self.ecg_df.shape[0]
        self.wave_len = self.ecg_df.shape[1] - 1
    
    def __len__(self):
        return self.df_len

    def get_X(self, idx):
        return self.ecg_df.iloc[idx].to_numpy()[:-1]
    
    def get_y(self, idx):
        return int(self.ecg_df.iloc[idx].to_numpy()[-1])

    def plot(self, idx):
        t = np.arange(self.wave_len) / self.sr
        plt.figure(figsize = (7, 7))
        plt.plot(t, self.get_X(idx))
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid()
        wave_dir = PROJECT_ROOT / 'ecg_wave'
        wave_dir.mkdir(parents = True, exist_ok = True)
        plt.savefig(wave_dir / f'{idx:05d}-{self.df_type}.png')
        plt.close()


class SCGGenerator:
    def __init__(self, df_type, sr = 125):
        self.sr = sr
        self.df_type = df_type
        self.digitizer = Digitizer(self.df_type)
        self.wave_len = self.digitizer.wave_len
    
    def __len__(self):
        return len(self.digitizer)

    def get_X(self, idx):
        cwtmat, frequencies = pywt.cwt(self.digitizer.get_X(idx), np.arange(1, 188), 'morl', sampling_period = 1 / self.sr)
        scalogram = np.abs(cwtmat)
        return scalogram, frequencies

    def get_y(self, idx):
        return self.digitizer.get_y(idx)
    
    def plot(self, idx):
        t = np.arange(self.wave_len) / self.sr
        scalogram, frequencies = self.get_X(idx)
        plt.figure(figsize = (8.5, 7))
        plt.imshow(scalogram, extent = [t.min(), t.max(), frequencies.min(), frequencies.max()], cmap = 'jet', aspect = 'auto', vmax = scalogram.max(), vmin = scalogram.min(), origin = 'lower')
        plt.colorbar(label = 'Power')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.grid(True, which = 'both', linestyle = ':', linewidth = 0.5)
        plt.tight_layout()
        scalogram_dir = PROJECT_ROOT / 'scalogram'
        scalogram_dir.mkdir(parents = True, exist_ok = True)
        plt.savefig(scalogram_dir / f'{idx:05d}-{self.df_type}.png')
        plt.close()
