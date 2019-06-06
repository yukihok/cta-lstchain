import numpy as np

class PulseIntegrator():
    
    def __init__(self, tel_id):
        
        self.tel_id = tel_id
        self.start_offset = 5
        self.slide_time = 10
        self.num_gains = 2
        self.num_pixels = 1855

    def sliding_window(self, event, slide_width, window_width):
        
        full_width = slide_width * 2 + 1
        waveforms = event.r1.tel[self.tel_id].waveform[:, :, 2:38]
        peakpos = np.argmax(waveforms, axis = 2)
        slide_start = peakpos - self.start_offset
        slide_time = 10
        ind = np.indices(waveforms.shape)[2]
        slide_max = np.zeros((self.num_gains, self.num_pixels))
        window_center = np.zeros((self.num_gains, self.num_pixels))
        for num_slide in range(0, slide_time):
            front = slide_start + num_slide
            end = front + full_width
            slide_window = (ind >= front[..., None]) & (ind < end[..., None])
            cur_slide_sum = np.sum(waveforms * slide_window, axis = 2)
            replaced = cur_slide_sum > slide_max
            window_center = window_center * ~replaced + (front + slide_width) * replaced
            slide_max = slide_max * ~replaced + cur_slide_sum * replaced

        window = (ind >= window_center[..., None] - window_width) & (ind <= window_center[..., None] + window_width)
        return window
        
