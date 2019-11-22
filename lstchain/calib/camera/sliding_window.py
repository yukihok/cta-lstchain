import numpy as np
import math
from numba import jit


class Integrators:
    
    def __init__(self, tel_id, data_type):
        
        self.tel_id = tel_id
        #self.slide_ite = 7
        self.num_gains = 2
        self.num_pixels = 1855
        self.data_type = data_type
        if self.data_type == 'real':
            self.start_slice = 2
            self.end_slice = 38
            self.ind = np.arange(self.start_slice, self.end_slice, 1)  # for real data
        if self.data_type == 'mc':
            self.ind = np.arange(0, 40, 1)  # for MC data

    @jit
    def set_slidecenter_around_peak(self, waveforms, full_width):

        oneside_width = (full_width - 1)/2
        peakpos = np.argmax(waveforms, axis = 2)
        slide_start = peakpos - self.start_offset
        slide_max = np.zeros((self.num_gains, self.num_pixels), dtype=np.int16)
        window_center = np.zeros((self.num_gains, self.num_pixels), dtype=np.int16)
        
        for num_slide in range(0, self.slide_ite):
            front = slide_start + num_slide
            end = front + full_width
            slide_window = (self.ind >= front[..., None]) & (self.ind < end[..., None])
            cur_slide_sum = np.sum(waveforms * slide_window, axis=2)
            to_be_replaced = cur_slide_sum > slide_max
            window_center = window_center * ~to_be_replaced + (front + oneside_width) * to_be_replaced
            slide_max = slide_max * ~to_be_replaced + cur_slide_sum * to_be_replaced

        return window_center
    
    @jit
    def set_trapezoid_slidecenter(self, waveforms, full_width, slide_start, ite):

        oneside_width = (full_width - 1)/2
        #peakpos = np.argmax(waveforms, axis = 2)
        #slide_start = peakpos - start_offset
        slide_max = np.zeros((self.num_gains, self.num_pixels), dtype=np.int16)
        window_center = np.zeros((self.num_gains, self.num_pixels), dtype=np.int16)
        
        for num_slide in range(ite):

            front = slide_start + num_slide
            end = front + full_width
            slide_window_internal = (self.ind > front[..., None]) & (self.ind < end[..., None])
            slide_window_edge = (self.ind == front[..., None]) & (self.ind == end[..., None])
            cur_slide_sum = np.sum(waveforms * slide_window_internal + waveforms * slide_window_edge / 2, axis=2)
            to_be_replaced = cur_slide_sum > slide_max
            window_center = window_center * ~to_be_replaced + (front + oneside_width) * to_be_replaced
            slide_max = slide_max * ~to_be_replaced + cur_slide_sum * to_be_replaced

        return window_center

    @jit
    def set_slidecenter_with_range(self, waveforms, full_width, start_cell, num_ite):
 
        oneside_width = (full_width - 1)/2
        slide_start = np.zeros((self.num_gains, self.num_pixels)) + start_cell
        slide_max = np.zeros((self.num_gains, self.num_pixels), dtype=np.int16)
        window_center = np.zeros((self.num_gains, self.num_pixels), dtype=np.int16)
        
        for num_slide in range(0, num_ite):
            front = slide_start + num_slide
            end = front + full_width
            slide_window = (self.ind >= front[..., None]) & (self.ind < end[..., None])
            cur_slide_sum = np.sum(waveforms * slide_window, axis=2)
            to_be_replaced = cur_slide_sum > slide_max
            window_center = window_center * ~to_be_replaced + (front + oneside_width) * to_be_replaced
            slide_max = slide_max * ~to_be_replaced + cur_slide_sum * to_be_replaced
            
        return window_center
        
    def set_window(self, center, fwidth, bwidth):

        window = (self.ind >= center[..., None] - fwidth) & (self.ind <= center[..., None] + bwidth)

        #if (np.any(center[..., None] - fwidth < 0) or np.any(center[..., None] + bwidth >= self.ind.shape[0])):
            #print('window is out of ROI!')
        
        return window

    def sliding_integration(self, waveforms, full_width):

        oneside_width = (full_width - 1)/2
        center = self.set_slidecenter_around_peak(waveforms, full_width)
        window = self.set_window(center, fwidth=oneside_width, bwidth=oneside_width)
        windowed = waveforms * window
        charge = np.sum(windowed, axis=2)

        return windowed, charge

    def effective_charge_diff(self, old_waveforms, sampling_interval, width, center, fwidth, bwidth):

        # trapezoidal integration is assumed.
        # calculate effective sampling interval
        inner_integral_time = np.sum(
            self.set_window(center, fwidth=fwidth - 1, bwidth=bwidth)
            * sampling_interval[:, :, self.start_slice:self.end_slice],
            axis=2)
        residual = inner_integral_time - (width - 2)  # approximation
        integral_time = inner_integral_time + residual

        # calculate charge difference to be corrected
        edge_window = np.logical_or((self.ind == center[..., None] - fwidth),
                                    (self.ind == center[..., None] + bwidth + 1))
        edge_charge = np.sum(old_waveforms * edge_window, axis=2)
        time_diff = integral_time - width
        diff_charge = edge_charge * (time_diff/2)
        
        return integral_time, diff_charge

    def trapezoid_integration(self, waveforms, center, fwidth, bwidth):

        # A bit complicated because 'center' deceided by set_trapezoid_slidecenter_around_peak
        # is not exactly center for trapezoid integration.
        # Exact center is between two samples...
        inner_window = self.set_window(center, fwidth - 1, bwidth)
        edge_window = np.logical_or((self.ind == center[..., None] - fwidth),
                                    (self.ind == center[..., None] + bwidth + 1))
        total_window = inner_window + edge_window
        inner_windowed = waveforms * inner_window
        edge_windowed = waveforms * edge_window
        total_windowed = inner_windowed + edge_windowed
        inner_sum = np.sum(inner_windowed, axis=2)
        edge_sum = np.sum(edge_windowed, axis=2)/2
        charge = inner_sum + edge_sum

        return total_window, charge

    def calc_peak_slice(self, waveforms, center, fwidth, bwidth):

        window = self.set_window(center = center, fwidth = fwidth, bwidth = bwidth)
        windowed = waveforms * window

        if np.any(np.sum(windowed, axis = 2) == 0):

            return False

        else:

            inds = np.zeros((self.num_gains, self.num_pixels, self.ind.shape[0]))
            inds[:, :] = self.ind
            peak_time = np.average(inds, axis = 2, weights = windowed)

            return peak_time

    def timing_integrate(self, waveforms, timing, width):

        half_width = (width-1) / 2
        left_edge = timing - half_width
        right_edge = timing + half_width

        left_inner = np.ceil(left_edge)
        right_inner = np.floor(right_edge)

        inner_window = (self.ind >= left_inner[..., None]) & (self.ind <= right_inner[..., None])
        inner_sum = np.sum(waveforms * inner_window, axis=2)

        left_outer_window = self.set_window(left_inner, 0, 0)
        left_outer_time = left_inner - left_edge
        left_outer_amp = np.sum(waveforms * left_outer_window, axis=2)
        left_outer = left_outer_amp * left_outer_time

        right_outer_window = self.set_window(right_inner, 0, 0)
        right_outer_time = right_inner - right_edge
        right_outer_amp = np.sum(waveforms * right_outer_window, axis=2)
        right_outer = right_outer_amp * right_outer_time

        charge = inner_sum + left_outer + right_outer
        window = inner_window + left_outer_window + right_outer_window

        return charge, window

    def subtract_offset(self, waveforms, shift):

            offset_width = 5
            peakpos = np.argmax(waveforms, axis = 2)
            shifted = peakpos + shift
            offset_window = self.set_window(shifted, 0, offset_width)
            windowed = waveforms * offset_window
            offset = np.sum(windowed, axis = 2)/ offset_width
            subtracted = waveforms - offset[:, :, np.newaxis]

            return subtracted

        
        
        
