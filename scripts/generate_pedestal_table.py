import argparse
import numpy as np
from astropy.io import fits
from numba import prange
import csv
from ctapipe_io_lst import LSTEventSource
from ctapipe.io import EventSeeker
from distutils.util import strtobool
from traitlets.config.loader import Config
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.drs4 import DragonPedestal
from lstchain.calib.camera.pedestal_correction import spike_judge

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument("--input_file", help="Path to fitz.fz file to create pedestal file.",
                    type=str, default="")

parser.add_argument("--mod", help="module id (0-264)",
                    type=int)

parser.add_argument("--gain", help="high gain : 0, low gain : 1",
                    type=int)

parser.add_argument("--pix", help="pixel number in a module (0-6)",
                    type=int)

# Optional argument
parser.add_argument("--max_events", help="Maximum numbers of events to read."
                                         "Default = 5000",
                    type=int, default=5000)

parser.add_argument("--output_file", help="Path where script create pedestal file",
                    type=str, default='')

args = parser.parse_args()

if __name__ == '__main__':
    print("input file: {}".format(args.input_file))
    print("max events: {}".format(args.max_events))
    reader = LSTEventSource(input_url=args.input_file, max_events=args.max_events)
    print("---> Number of files", reader.multi_file.num_inputs())


    mod = args.mod
    gain = args.gain
    pix = args.pix
    run_number = args.input_file.split('.')[2]
    output_filename = 'PedestalTable_{}_mod{}_gain{}_pix{}_{}eve.dat'.format(run_number, mod, gain, pix, args.max_events)
    if args.output_file != '':
        output_filename = args.output_file
    print('output file', output_filename)
    seeker = EventSeeker(reader)
    ev = seeker[0]
    tel_id = ev.r0.tels_with_data[0]
    n_modules = ev.lst.tel[tel_id].svc.num_modules
    expected_pixel_id = ev.lst.tel[tel_id].svc.pixel_ids
    size4drs = 4096

    config = Config({
        "LSTR0Corrections": {
            "tel_id": tel_id
        }
    })
    lst_r0 = LSTR0Corrections(config=config)
    pedestal = DragonPedestal(tel_id=tel_id, n_module=n_modules)
    last_time_read = np.zeros(size4drs)

    for event in reader:

        pixel = expected_pixel_id[mod * 7 + pix]
        samples = event.r0.tel[tel_id].waveform[gain, pixel]
        cur_first_cap = lst_r0._get_first_capacitor(event, mod)[gain, pix]
        cur_local_counter = event.lst.tel[tel_id].evt.local_clock_counter[mod]
        roisize = 40

        '''
        if event.count == 0:
            old_first_cap = cur_first_cap
            continue

        for nr_module in prange(0, self.n_module):
            pedestal.first_cap_array[nr_module, :, :] = pedestal.get_first_capacitor(event, nr_module)

        roisize = 40
        for nr_module in prange(0, n_module):
            first_cap = pedestal.first_cap_array[nr_module, :, :]
            for gain in prange(0, 2):
                for pix in prange(0, 7):
                    fc = first_cap[gain, pix]
                    pixel = expected_pixel_id[nr_module * 7 + pix]

                    posads0 = int((2 + fc) % size4drs)
                    if posads0 + 40 < size4drs:
                        meanped[gain, pixel, posads0:(posads0 + 36)] += waveform[gain, pixel, 2:38]
                        numped[gain, pixel, posads0:(posads0 + 36)] += 1

                    else:
                        for k in prange(2, roisize - 2):
                            posads = int((k + fc) % size4drs)
                            val = waveform[gain, pixel, k]
                            meanped[gain, pixel, posads] += val
                            numped[gain, pixel, posads] += 1

        '''

        for icell in prange(2, roisize-2):
            cap_id = int((cur_first_cap + icell) % size4drs)

            #if spike_judge(pix, old_first_cap, cap_id):  # discard events containing spikes
                #break

            if last_time_read[cap_id] > 0:
                time_diff = cur_local_counter - last_time_read[cap_id]
                time_diff_ms = time_diff / (133.e3)
                '''
                if time_diff_ms > 100:
                    pedestal.meanped[gain, pix, cap_id] += samples[icell]
                    pedestal.numped[gain, pix, cap_id] += 1
                '''
                # same algorithm as Pawel's
                if time_diff_ms < 100:
                    #samples[icell] = samples[icell] - (23.03 * np.power(time_diff_ms, -0.25) - 9.73)
                    samples[icell] = samples[icell] - (27.33 * np.power(time_diff_ms, -0.24) - 10.4)

            pedestal.meanped[gain, pix, cap_id] += samples[icell]
            pedestal.numped[gain, pix, cap_id] += 1

        old_first_cap = cur_first_cap
        for icell in prange(0, roisize-1):
            cap_id = int((cur_first_cap + icell) % size4drs)
            last_time_read[cap_id] = cur_local_counter

        if event.r0.event_id % 1000 == 0:
            print("event id = {}".format(event.r0.event_id))

    # Finalize pedestal and write to fits file
    print(pedestal.numped[gain, pix])
    meanped = pedestal.meanped[gain, pix]/pedestal.numped[gain, pix]

    output_file = open(output_filename, 'w')
    writer = csv.writer(output_file)
    for icap in range(size4drs):
        writer.writerow([str(pedestal.numped[gain, pix, icap]), str(meanped[icap])])
