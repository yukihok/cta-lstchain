import numpy as np
from numba import jit, prange


@jit(parallel=True)
def calc_dt(event, lst_r0, mod, gain, pix, last_time_read, charge_list, dt_list):
    
    tel_id = 0
    size4drs = 4096
        
    time_now = event.lst.tel[tel_id].evt.local_clock_counter[mod]
    fc = lst_r0._get_first_capacitor(event, mod)[gain, pix]
    expected_pixel_id = event.lst.tel[tel_id].svc.pixel_ids
    
    pixel = expected_pixel_id[mod*7 + pix]

    for icell in prange(0, 40):
        cap_id = int((icell + fc) % size4drs)

        last_time_read[mod, gain, pix, cap_id]
        if last_time_read[mod, gain, pix, cap_id] > 0:
            time_diff = time_now - last_time_read[mod, gain, pix, cap_id]
            time_diff_ms = time_diff / (133e3)

            charge_list.append(event.r1.tel[tel_id].waveform[gain, pix, icell])
            dt_list.append(time_diff_ms)
            #print(charge_list)
            #print(dt_list)
            
    first_cap = int(fc)

    if first_cap + 40 < 4096:
        last_time_read[mod, gain, pix, first_cap:(first_cap + 39)] = time_now[..., None]
    else:
        for icell in prange(0, 39):
            cap_id = int((icell + fc) % size4drs)
            last_time_read[mod, gain, pix, cap_id] = time_now

    # now the magic of Dragon,
    # if the ROI is in the last quarter of each DRS4
    # for even channel numbers extra 12 slices are read in a different place
    # code from Takayuki & Julian
    if pix % 2 == 0:
        first_cap = fc

        if first_cap % 1024 > 766 and first_cap % 1024 < 1012:
            start = int(first_cap) + 1024 - 1
            end = int(first_cap) + 1024 + 11
            last_time_read[mod, gain, pix, start%4096:end%4096] = time_now[..., None]

        elif first_cap % 1024 >= 1012:
            ring = int(first_cap / 1024)
            for cell in range(first_cap + 1024, (ring + 2) * 1024):
                last_time_read[mod, gain, pix, int(cell) % 4096] = time_now[..., None]


def spike_judge(event, old_first_cap, mod, gain, pix, cap_id):

    roisize = 40
    size4drs = 4096
    old_finish_cap = (old_first_cap + roisize - 1)%size4drs
    current_first_cap = lst_r0._get_first_capacitor(event, mod)[gain, pix]

    if judge_spike_A(old_finish_cap, pix, cap_id):
        return True

    
def even_channel(pix):

    if pix in [0, 1, 4, 5]:
            return True
    else:
        return False

def judge_spike_A(old_finish_cap, pix, cap_id):

    size4drs = 4096
    size1drs = 1024
    if even_channel(pix):
        if old_finish_cap % size4drs < size1drs/2 and cap_id % size4drs < size1drs/2:
            if (cap_id % size1drs == old_finish_cap % size1drs) or (cap_id % size1drs == (old_finish_cap - 1)% size1drs):
                return True

def judge_spike_B(old_finish_cap, pix, cap_id):

    size4drs = 4096
    size1drs = 1024
    if even_channel(pix):
        if old_finish_cap % size4drs < size1drs/2 and cap_id % size4drs > size1drs/2:
            if (cap_id % size1drs == 1021 - old_finish_cap % size1drs) or (cap_id % size1drs == 1022 - old_finish_cap % size1drs):
                return True

def judge_spike_C(old_first_cap, cap_id):

    size4drs = 4096
    size1drs = 1024
    if cap_id % size1drs == old_first_cap % size1drs - 1:
        return True