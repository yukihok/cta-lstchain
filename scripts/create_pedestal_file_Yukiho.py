import argparse
import numpy as np
from astropy.io import fits
from numba import jit, prange
from ctapipe_io_lst import LSTEventSource
from ctapipe.io import EventSeeker
from distutils.util import strtobool
from traitlets.config.loader import Config
<<<<<<< HEAD
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.drs4 import DragonPedestal

=======
from lstchain.calib.camera.r0_Yukiho import LSTR0Corrections
from lstchain.calib.camera.drs4_Yukiho import DragonPedestal
from lstchain.calib.camera.pedestal_correction import spike_judge, spike_judge_bycap
>>>>>>> 83386c62f8d504e4a3c2497a1ffd4fd232f1b22f

''' 
Script to create pedestal file for low level calibration. 
To run script in console:
python create_pedestal_file.py --input_file LST-1.1.Run00097.0000.fits.fz --output_file pedestal.fits 
--max_events 9000
not to use deltaT correction add --deltaT False
'''

<<<<<<< HEAD
@jit
def fill_pedestal_without_dt_corr(waveform, expected_pixel_id, local_clock_list, fc, last_time_array, meanped, numped):

    num_modules = 265
    size4drs = 4096
=======
@jit(parallel=True)
def ped_time(timediff):
    """
    Power law function for time lapse baseline correction.
    Coefficients from curve fitting to dragon test data
    at temperature 30 degC
    """
    return 27.33 * np.power(timediff, -0.24) - 10.4
    #return (23.03 * np.power(timediff, -0.25) - 9.73)  # for 40degC


@jit
def fill_pedestal_without_dt_corr(waveform, expected_pixel_id, local_clock_list, old_fc, fc, last_time_array, meanped, numped, spikes):

    num_modules = 265
    size4drs = 4096

>>>>>>> 83386c62f8d504e4a3c2497a1ffd4fd232f1b22f
    for nr_module in prange(0, num_modules):
        time_now = local_clock_list[nr_module]
        for gain in prange(0, 2):
            for pix in prange(0, 7):
                pixel = expected_pixel_id[nr_module * 7 + pix]
                first_cap = int(fc[nr_module, gain, pix])
<<<<<<< HEAD
                for cell in prange(0, 40):
                    cap_id = int((cell + first_cap) % size4drs)
                    if last_time_array[nr_module, gain, pix, cap_id] > 0:
                        time_diff = time_now - last_time_array[nr_module, gain, pix, cap_id]
                        time_diff_ms = time_diff / (133.e3)
=======
                #old_first_cap = old_fc[nr_module, gain, pix]
                #if spike_judge(pix, old_first_cap, first_cap) != 0:
                    #print('spike!')
                    #spikes[nr_module, gain, pix] += 1
                    #break
                for cell in prange(2, 38):
                    cap_id = int((cell + first_cap) % size4drs)
                    #if spike_judge_bycap(pix, old_first_cap, cap_id) != 0:
                        #break
                    if last_time_array[nr_module, gain, pix, cap_id] > 0:
                        time_diff = time_now - last_time_array[nr_module, gain, pix, cap_id]
                        time_diff_ms = time_diff / (133.e3)

                        # my algorithm
>>>>>>> 83386c62f8d504e4a3c2497a1ffd4fd232f1b22f
                        if time_diff_ms > 100:
                            val = waveform[gain, pixel, cell]
                            meanped[gain, pixel, cap_id] += val
                            numped[gain, pixel, cap_id] += 1

<<<<<<< HEAD
=======
                    '''
                        if time_diff_ms < 100:
                            #samples[icell] = samples[icell] - (23.03 * np.power(time_diff_ms, -0.25) - 9.73)
                            waveform[gain, pixel, cell] -= ped_time(time_diff_ms)

                    meanped[gain, pixel, cap_id] += waveform[gain, pixel, cell]
                    numped[gain, pixel, cap_id] += 1
                    '''

>>>>>>> 83386c62f8d504e4a3c2497a1ffd4fd232f1b22f
                if first_cap + 40 < 4096:
                    last_time_array[nr_module, gain, pix, first_cap:(first_cap + 39)] = time_now
                else:
                    for cell in prange(0, 39):
                        cap_id = int((cell + first_cap) % size4drs)
                        last_time_array[nr_module, gain, pix, cap_id] = time_now

                # now the magic of Dragon,
                # if the ROI is in the last quarter of each DRS4
                # for even channel numbers extra 12 slices are read in a different place
                # code from Takayuki & Julian
                if pix % 2 == 0:
                    if first_cap % 1024 > 766 and first_cap % 1024 < 1012:
                        start = int(first_cap) + 1024 - 1
                        end = int(first_cap) + 1024 + 11
                        last_time_array[nr_module, gain, pix, start%4096:end%4096] = time_now
                    elif first_cap % 1024 >= 1012:
                        channel = int(first_cap / 1024)
                        for kk in range(first_cap + 1024, (channel + 2) * 1024):
                            last_time_array[nr_module, gain, pix, int(kk) % 4096] = time_now

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument("--input_file", help="Path to fitz.fz file to create pedestal file.",
                    type=str, default="")

parser.add_argument("--output_file", help="Path where script create pedestal file",
                    type=str)

# Optional argument
parser.add_argument("--max_events", help="Maximum numbers of events to read."
                                         "Default = 5000",
                    type=int, default=5000)

parser.add_argument('--deltaT', '-s', type=lambda x: bool(strtobool(x)),
                    help='Boolean. True for use deltaT correction'
                    'Default=True, use False otherwise',
                    default=True)

args = parser.parse_args()

if __name__ == '__main__':
    print("input file: {}".format(args.input_file))
    print("max events: {}".format(args.max_events))
    reader = LSTEventSource(input_url=args.input_file, max_events=args.max_events)
    print("---> Number of files", reader.multi_file.num_inputs())

    seeker = EventSeeker(reader)
    ev = seeker[0]
    tel_id = ev.r0.tels_with_data[0]
    n_modules = ev.lst.tel[tel_id].svc.num_modules
    expected_pixel_id = ev.lst.tel[tel_id].svc.pixel_ids
    roisize = 40

    config = Config({
        "LSTR0Corrections": {
            "tel_id": tel_id
        }
    })
    lst_r0 = LSTR0Corrections(config=config)
    pedestal = DragonPedestal(tel_id=tel_id, n_module=n_modules)
    last_time_read = lst_r0.last_reading_time_array

    if args.deltaT:
        print("DeltaT correction active")
        for i, event in enumerate(reader):
            lst_r0.time_lapse_corr(event)
            pedestal.fill_pedestal_event(event)
            if i%500 == 0:
                print("i = {}, ev id = {}".format(i, event.r0.event_id))

    else:
        print("DeltaT correction no active")
<<<<<<< HEAD
=======
        spikes = np.zeros((265, 2, 7))
>>>>>>> 83386c62f8d504e4a3c2497a1ffd4fd232f1b22f
        for event in reader:

            for nr_module in prange(0, n_modules):
                pedestal.first_cap_array[nr_module, :, :] = pedestal.get_first_capacitor(event, nr_module)

<<<<<<< HEAD
            samples = event.r0.tel[tel_id].waveform
            fc = pedestal.first_cap_array
            cur_local_counter = event.lst.tel[tel_id].evt.local_clock_counter
            fill_pedestal_without_dt_corr(samples, expected_pixel_id, cur_local_counter, fc, last_time_read, pedestal.meanped, pedestal.numped)

            if event.r0.event_id % 1000 == 0:
=======
            if event.count == 0:
                old_fc = pedestal.first_cap_array

            samples = event.r0.tel[tel_id].waveform
            fc = pedestal.first_cap_array
            cur_local_counter = event.lst.tel[tel_id].evt.local_clock_counter
            fill_pedestal_without_dt_corr(samples, expected_pixel_id, cur_local_counter, old_fc, fc, last_time_read, pedestal.meanped, pedestal.numped, spikes)
            old_fc = fc

            if event.r0.event_id % 500 == 0:
>>>>>>> 83386c62f8d504e4a3c2497a1ffd4fd232f1b22f
                print("ev id = {}".format(event.r0.event_id))


    # Finalize pedestal and write to fits file
    print(np.min(pedestal.numped))
    print(pedestal.numped)
<<<<<<< HEAD
    pedestal.finalize_pedestal()

    primaryhdu = fits.PrimaryHDU(ev.lst.tel[tel_id].svc.pixel_ids)
    secondhdu = fits.ImageHDU(np.int16(pedestal.meanped))

    hdulist = fits.HDUList([primaryhdu, secondhdu])
    hdulist.writeto(args.output_file)
=======
    print('spike_total', np.sum(spikes), 'spike max', np.max(spikes))
    pedestal.finalize_pedestal()

    primaryhdu = fits.PrimaryHDU(ev.lst.tel[tel_id].svc.pixel_ids)
    secondhdu = fits.ImageHDU(pedestal.meanped)

    hdulist = fits.HDUList([primaryhdu, secondhdu])
    hdulist.writeto(args.output_file)
>>>>>>> 83386c62f8d504e4a3c2497a1ffd4fd232f1b22f
