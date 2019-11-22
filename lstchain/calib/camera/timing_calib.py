import numpy as np
import pandas as pd
from numba import jit, prange


class DRSTimingCalibrator:

    def __init__(self, tel_id, table_path):

        self.tel_id = tel_id
        self.ringsize = 1024
        self.roisize = 40
        self.num_gains = 2
        self.num_pixels = 1855
        self.spiral_first_caps = np.zeros((self.num_gains, self.num_pixels))
        # 1e6 is added temporalily to avoid samping interval being zero.
        self.sample_interval = pd.read_pickle(table_path) + 1e-6

        # don't use pixels with no TP data
        self.exceptional_pixels = [206,207,258,259,260,317,318,342,343,409,410,411,483,484,696,697,790,791,792,891,892,1502,1503,1616,1617,1618,1713,1714]
        for ipix in self.exceptional_pixels:
            self.sample_interval[:, ipix, :] = np.ones((self.num_gains, self.ringsize))
        '''
        # in case of reading text file
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
        '''

    @jit
    def calc_sample_interval(self, event, first_caps):

        n_pix = 7
        n_mod = 265
        expected_pixel_id = event.lst.tel[self.tel_id].svc.pixel_ids
        for imod in prange(n_mod):
            for ipix in prange(n_pix):
                hard_pix = imod * n_pix + ipix
                spiral_pix = expected_pixel_id[hard_pix]
                self.spiral_first_caps[:, spiral_pix] = first_caps[imod][:, ipix]

        mod_first_caps = np.array(self.spiral_first_caps % self.ringsize, dtype='int16')
        ind = np.arange(0, self.ringsize, 1)
        caps_in_roi = np.logical_and(
            (ind >= mod_first_caps[..., None]), (ind < mod_first_caps[..., None] + self.roisize))
        residual = mod_first_caps + self.roisize - self.ringsize
        residual_caps_in_roi = ind < residual[..., None]
        total_caps_in_roi = caps_in_roi + residual_caps_in_roi
        sample_interval_in_roi = self.sample_interval * total_caps_in_roi
        sample_interval_in_roi = \
            sample_interval_in_roi[sample_interval_in_roi > 0].reshape(
                self.num_gains, self.num_pixels, self.roisize
            )
        
        return sample_interval_in_roi




        
