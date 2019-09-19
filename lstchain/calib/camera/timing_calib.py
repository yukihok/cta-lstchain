import numpy as np
import pandas as pd
from numba import prange


class DRSTimingCalibrator:

    def __init__(self, tel_id, table_path):

        self.tel_id = tel_id
        self.ringsize = 1024
        self.roisize = 40
        self.num_gains = 2
        self.num_pixels = 1855
        self.spiral_first_caps = np.zeros((self.num_gains, self.num_pixels))
        self.sample_interval = np.ones((self.num_gains, self.num_pixels, self.ringsize))
        df = pd.read_csv(table_path, sep=' ')

        for igain in range(self.num_gains):

            if igain == 0:
                gain_str = 'H'
            else:
                gain_str = 'L'

            for ipix in prange(self.num_pixels):
                column = '{}{}'.format(ipix, gain_str)
                if column in df.columns:
                    dt_samples = df[[column]].values.reshape(self.ringsize)
                    self.sample_interval[igain, ipix, :] = dt_samples

    def reorder_first_caps(self, event, first_caps, mod):

        n_pix = 7
        expected_pixel_id = event.lst.tel[self.tel_id].svc.pixel_ids
        for ipix in prange(n_pix):
            hard_pix = mod * n_pix + ipix
            spiral_pix = expected_pixel_id[hard_pix]
            self.spiral_first_caps[:, spiral_pix] = first_caps[mod][:, ipix]

    def calibrate(self, event):

        mod_first_caps = np.array(self.spiral_first_caps % self.ringsize, dtype='int16')
        ind = np.arange(0, self.ringsize, 1)
        caps_in_roi = np.logical_and(
            (ind >= mod_first_caps[..., None]), (ind < mod_first_caps[..., None] + self.roisize))
        residual = mod_first_caps + self.roisize - self.ringsize
        residual_caps_in_roi = ind < residual[..., None]
        total_caps_in_roi = caps_in_roi + residual_caps_in_roi
        sample_interval_in_roi = self.sample_interval * total_caps_in_roi
        sample_interval_in_roi = \
            sample_interval_in_roi[sample_interval_in_roi > 0].reshape(self.num_gains, self.num_pixels, self.roisize)
        event.r1.tel[self.tel_id].waveform = event.r1.tel[self.tel_id].waveform * sample_interval_in_roi
