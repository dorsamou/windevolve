# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/test_data.py

Description:
    copy of the official SDWPF per-window loader. Wraps ONE
    window CSV (all 134 turbines stacked turbine-major) and hands back
    per-turbine slices. Keeps two copies of the data: a model-facing copy with
    NaN replaced by 0 and only the modeled columns, and a mask-facing copy with
    NaNs preserved and all columns.

Citation:
    official Baidu KDD Cup 2022 SDWPF evaluation code:
      PaddlePaddle/PaddleSpatial,
      apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/paddlepaddle/test_data.py
      https://github.com/PaddlePaddle/PaddleSpatial  (accessed 2026-07-22)
    Dataset / methodology:
      Zhou et al. "SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting
      Challenge at KDD Cup 2022." arXiv:2208.04360, 2022.
      Zhou et al. "A Dataset for Spatial Dynamic Wind Power Forecasting over a
      Large Turbine Array." Scientific Data 11:649, 2024.
      doi:10.1038/s41597-024-03427-5
"""
import numpy as np                # array conversion / NaN replacement
import pandas as pd               # CSV reading and column handling
from copy import deepcopy         # independent NaN-filled copy of the frame

# Silence pandas' SettingWithCopyWarning triggered by the in-place NaN fill.
pd.set_option('mode.chained_assignment', None)


class TestData(object):
    """
    Description:
        Loader for a single SDWPF window CSV (all turbines stacked). Splits the
        flat table into per-turbine blocks and exposes both a model-ready array
        and a mask-ready raw frame for each turbine.
    """
    def __init__(self,
                 path_to_data,
                 task='MS',
                 target='Patv',
                 start_col=3,
                 farm_capacity=134):
        """
        Description:
            Read the window CSV and prepare the two data copies.
        Args:
            path_to_data: path to one window CSV (infile or outfile)
            task: prediction task type; only 'MS' (multi-in, single-out) is used
            target: name of the target column (stored; selection is positional)
            start_col: index of the first modeled column (3 -> Wspd; 0-2 are labels)
            farm_capacity: number of turbines stacked in the file (134)
        Returns:
            None
        """
        self.task = task                                  # task type flag
        self.target = target                              # target column name
        self.start_col = start_col                        # first modeled column index
        self.data_path = path_to_data                     # source CSV path
        self.farm_capacity = farm_capacity                # number of turbines (134)
        self.df_raw = pd.read_csv(self.data_path)         # raw frame, NaNs preserved
        self.total_size = int(self.df_raw.shape[0] / self.farm_capacity)  # rows per turbine
        # Model-facing copy: identical rows but with NaN replaced by 0.
        self.df_data = deepcopy(self.df_raw)
        self.df_data.replace(to_replace=np.nan, value=0, inplace=True)

    def get_turbine(self, tid):
        """
        Description:
            Return one turbine's model array and raw frame for this window.
        Args:
            tid: turbine index in [0, farm_capacity)
        Returns:
            A tuple (seq, df): seq is the NaN-filled modeled-columns array
            (shape (total_size, n_modeled_cols)); df is the raw frame slice
            (all columns, NaNs kept).
        """
        # Turbine-major slicing: turbine tid occupies a contiguous row block.
        begin_pos = tid * self.total_size
        border1 = begin_pos
        border2 = begin_pos + self.total_size
        if self.task == 'MS':
            cols = self.df_data.columns[self.start_col:]  # drop the 3 label columns
            data = self.df_data[cols]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        seq = data.values[border1:border2]                # model array (NaN->0)
        df = self.df_raw[border1:border2]                 # raw frame (NaNs kept)
        return seq, df

    def get_all_turbines(self):
        """
        Description:
            Slice the whole window into per-turbine pieces.
        Args:
            None
        Returns:
            A tuple (seqs, dfs): lists of length farm_capacity holding each
            turbine's model array and raw frame respectively.
        """
        seqs, dfs = [], []
        for i in range(self.farm_capacity):
            seq, df = self.get_turbine(i)
            seqs.append(seq)
            dfs.append(df)
        return seqs, dfs
