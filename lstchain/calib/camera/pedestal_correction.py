import numpy as np
from numba import jit, prange


@jit(parallel=True)
def calc_dt(event, lst_r0, mod, gain, pix, last_time_read, charge_list, dt_list):
    
    tel_id = 0
    size4drs = 4096
        
    time_now = event.lst.tel[tel_id].evt.local_clock_counter[mod]
    fc = lst_r0._get_first_capacitor(event, mod)[gain, pix]
    expected_pixel_id = event.lst.tel[tel_id].svc.pixel_ids
    current_dt = []
    
    pixel = expected_pixel_id[mod*7 + pix]

    for icell in prange(2, 38):
        cap_id = int((icell + fc) % size4drs)

        last_time_read[mod, gain, pix, cap_id]
        if last_time_read[mod, gain, pix, cap_id] > 0:
            time_diff = time_now - last_time_read[mod, gain, pix, cap_id]
            time_diff_ms = time_diff / (133e3)

            charge_list.append(event.r1.tel[tel_id].waveform[gain, pix, icell])
            dt_list.append(time_diff_ms)
            current_dt.append(time_diff_ms)
            #print(charge_list)
            #print(dt_list)
            
    first_cap = int(fc)

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

    return current_dt

@jit(parallel=True)
def spike_judge(old_first_cap, first_cap):

    roisize = 40
    size1drs = 1024
    spike = -1
    old_finish_pos = int((old_first_cap + roisize - 1)%size1drs)
    first_cell = int(first_cap % size1drs)
    channel = int(first_cap//size1drs + 1)
    pos = -1

    spikepos_A1 = old_finish_pos  # pattern 1
    spikepos_A2 = int((old_finish_pos - 1)%size1drs)  # pattern 2

    if (first_cell + 2)%size1drs < spikepos_A1 and (first_cell + 38)%size1drs > spikepos_A1:
        spike_cell = int((spikepos_A1 - first_cell)%size1drs)
        if old_finish_pos < 512:
            spike = 1
            pos = spike_cell
    if (first_cell + 2)%size1drs < spikepos_A2 and (first_cell + 38)%size1drs > spikepos_A2:
        spike_cell = int((spikepos_A2 - first_cell)%size1drs)
        if old_finish_pos < 512 and spikepos_A2 < 512:
            spike = 1
            pos = spike_cell


    spikepos_B1 = int((1021 - old_finish_pos)%size1drs)
    spikepos_B2 = int((1022 - old_finish_pos)%size1drs)
    if (first_cell + 2)%size1drs <= spikepos_B1 and (first_cell + 38)%size1drs > spikepos_B1:
        spike_cell = int((spikepos_B1 - first_cell) % size1drs)
        if old_finish_pos < 512 and spikepos_B1 > 512:
            spike = 2
            pos = spike_cell
    if (first_cell + 2)%size1drs <= spikepos_B2 and (first_cell + 38)%size1drs > spikepos_B2:
        spike_cell = int((spikepos_B2 - first_cell) % size1drs)
        if old_finish_pos < 512 and spikepos_B2 > 512:
            spike = 2
            pos = spike_cell

    spikepos_C = int((old_first_cap - 1)%size1drs)
    if (first_cell + 2)%size1drs <= spikepos_C and (first_cell + 38)%size1drs > spikepos_C:
        spike_cell = int((spikepos_C - first_cell) % size1drs)
        spike = 3
        pos = spike_cell

    return spike, pos

@jit(parallel=True)
def spike_judge_bycap(pix, old_first_cap, cap_id):

    roisize = 40
    size1drs = 1024
    old_finish_pos = int((old_first_cap + roisize - 1)%size1drs)
    old_finish_pos_next = int((old_finish_pos - 1) % size1drs)
    spikepos = int(cap_id % size1drs)
    spike = -1

    if old_finish_pos < 512 and spikepos < 512:
        if (spikepos == old_finish_pos) or (spikepos == old_finish_pos_next):
            spike = 1
    if old_finish_pos < 512 and spikepos > 512:
        if (spikepos == old_finish_pos) or (spikepos == old_finish_pos_next):
            spike = 2
    if spikepos == (old_first_cap - 1) % size1drs:
        spike = 3

    return spike

def even_channel(pix):

    if pix in [0, 1, 4, 5]:
        return True
    else:
        return False

@jit
def judge_spike_A(old_finish_cap, pix, cap_id):

    size4drs = 4096
    size1drs = 1024
    if even_channel(pix):
        if old_finish_cap % size1drs < size1drs/2 and cap_id % size1drs < size1drs/2:
            if (cap_id % size1drs == old_finish_cap % size1drs) or (cap_id % size1drs == (old_finish_cap - 1)% size1drs):
                return True

@jit
def judge_spike_B(old_finish_cap, pix, cap_id):

    size4drs = 4096
    size1drs = 1024
    if even_channel(pix):
        if old_finish_cap % size1drs < size1drs/2 and cap_id % size1drs > size1drs/2:
            if (cap_id % size1drs == 1021 - old_finish_cap % size1drs) or (cap_id % size1drs == 1022 - old_finish_cap % size1drs):
                return True

@jit
def judge_spike_C(old_first_cap, cap_id):

    size4drs = 4096
    size1drs = 1024
    if cap_id % size1drs == old_first_cap % size1drs - 1:
        return True