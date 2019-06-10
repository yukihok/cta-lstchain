import numpy as np

class SlidingWindow:
    
    def __init__(self, tel_id):
        
        self.tel_id = tel_id
        self.start_offset = 5
        self.slide_ite = 10
        self.num_gains = 2
        self.num_pixels = 1855
        self.ind = np.arange(2, 38, 1)

    def set_center(self, waveforms, oneside_width):

        full_width = oneside_width * 2 + 1
        peakpos = np.argmax(waveforms, axis = 2)
        slide_start = peakpos - self.start_offset
        slide_max = np.zeros((self.num_gains, self.num_pixels))
        window_center = np.zeros((self.num_gains, self.num_pixels))
        for num_slide in range(0, self.slide_ite):
            front = slide_start + num_slide
            end = front + full_width
            slide_window = (self.ind >= front[..., None]) & (self.ind < end[..., None])
            cur_slide_sum = np.sum(waveforms * slide_window, axis = 2)
            to_be_replaced = cur_slide_sum > slide_max
            window_center = window_center * ~to_be_replaced + (front + oneside_width) * to_be_replaced
            slide_max = slide_max * ~to_be_replaced + cur_slide_sum * to_be_replaced

        return window_center

    def set_window(self, center, width):

        window = (self.ind >= center[..., None] - width) & (self.ind <= center[..., None] + width)

        return window

    def calc_peak_slice(waveforms, center, width):

        region = set_window(center, width)
        windowed = waveforms * region
        peak_slice = np.average(ind, axis = 2, weights = windowed)

        return peak_slice

        
        
        
