
import numpy as np
from logic.pulsed.pulse_objects import PulseBlock, PulseBlockEnsemble, PulseSequence
from logic.pulsed.pulse_objects import PredefinedGeneratorBase

class LeeGoldburgGenerator(PredefinedGeneratorBase):
    """

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_FSLG(self, name='FSLG', nucl_rabi_period = 5*1e-6, nucl_larmor_freq = 1.8*1e+6,
                                 num_of_points = 40, xy16_order=2, rf_amp=0.25, alternating=True):
        """
        Generate Frequency Switched Lee Goldburg decoupling. RF amplitude must be as for the nuclear Rabi
        """
        created_blocks = list()
        created_ensembles = list()
        created_sequences = list()

        nucl_rabi_freq = 1.0/nucl_rabi_period

        theta = np.arccos(1/np.sqrt(3)) # magic angle
        radiofreq_1 = nucl_larmor_freq - nucl_rabi_freq/np.tan(theta)
        radiofreq_2 = nucl_larmor_freq + nucl_rabi_freq/np.tan(theta)

        dip_position = 1 / (2 * radiofreq_1)
        dip_position2 = 1 / (2 * radiofreq_2)

        tau_step = 1e-9
        tau_start = dip_position - tau_step * num_of_points / 2
        tau_start2 = dip_position2 - tau_step * num_of_points / 2

        # get tau array for measurement ticks
        tau_array = tau_start + np.arange(num_of_points) * tau_step
        tau_array2 = tau_start2 + np.arange(num_of_points) * tau_step

        # create the static waveform elements
        waiting_element = self._get_idle_element(length=self.wait_time, increment=0)
        laser_element = self._get_laser_element(length=self.laser_length, increment=0)
        delay_element = self._get_delay_element()

        pihalf_element_rf1 = self._get_mw_rf_element(length=self.rabi_period / 4,
                                                 increment=0,
                                                 amp1=self.microwave_amplitude,
                                                 freq1=self.microwave_frequency,
                                                 phase1=0,
                                                 amp2=rf_amp,
                                                 freq2=radiofreq_1,
                                                 phase2=0)

        last_pihalf_element_rf1 = self._get_mw_rf_gate_element(length=self.rabi_period / 4,
                                                           increment=0,
                                                           amp1=self.microwave_amplitude,
                                                           freq1=self.microwave_frequency,
                                                           phase1=0,
                                                           amp2=rf_amp,
                                                           freq2=radiofreq_1,
                                                           phase2=0)

        pi_x_element_rf1 = self._get_mw_rf_element(length=self.rabi_period / 2,
                                               increment=0,
                                               amp1=self.microwave_amplitude,
                                               freq1=self.microwave_frequency,
                                               phase1=0,
                                               amp2=rf_amp,
                                               freq2=radiofreq_1,
                                               phase2=0)

        pi_y_element_rf1 = self._get_mw_rf_element(length=self.rabi_period / 2,
                                               increment=0,
                                               amp1=self.microwave_amplitude,
                                               freq1=self.microwave_frequency,
                                               phase1=90.0,
                                               amp2=rf_amp,
                                               freq2=radiofreq_1,
                                               phase2=0)

        pi_minus_x_element_rf1 = self._get_mw_rf_element(length=self.rabi_period / 2,
                                                     increment=0,
                                                     amp1=self.microwave_amplitude,
                                                     freq1=self.microwave_frequency,
                                                     phase1=180.0,
                                                     amp2=rf_amp,
                                                     freq2=radiofreq_1,
                                                     phase2=0)

        pi_minus_y_element_rf1 = self._get_mw_rf_element(length=self.rabi_period / 2,
                                                     increment=0,
                                                     amp1=self.microwave_amplitude,
                                                     freq1=self.microwave_frequency,
                                                     phase1=270.0,
                                                     amp2=rf_amp,
                                                     freq2=radiofreq_1,
                                                     phase2=0)

        pi3half_element_rf1 = self._get_mw_rf_gate_element(length=self.rabi_period / 4,
                                                       increment=0,
                                                       amp1=self.microwave_amplitude,
                                                       freq1=self.microwave_frequency,
                                                       phase1=180.0,
                                                       amp2=rf_amp,
                                                       freq2=radiofreq_1,
                                                       phase2=0)

        # pulses for second RF frequency--------------------------------------------------------------------------------

        pihalf_element_rf2 = self._get_mw_rf_element(length=self.rabi_period / 4,
                                                     increment=0,
                                                     amp1=self.microwave_amplitude,
                                                     freq1=self.microwave_frequency,
                                                     phase1=0,
                                                     amp2=rf_amp,
                                                     freq2=radiofreq_2,
                                                     phase2=180)

        last_pihalf_element_rf2 = self._get_mw_rf_gate_element(length=self.rabi_period / 4,
                                                               increment=0,
                                                               amp1=self.microwave_amplitude,
                                                               freq1=self.microwave_frequency,
                                                               phase1=0,
                                                               amp2=rf_amp,
                                                               freq2=radiofreq_2,
                                                               phase2=180)

        pi_x_element_rf2 = self._get_mw_rf_element(length=self.rabi_period / 2,
                                                   increment=0,
                                                   amp1=self.microwave_amplitude,
                                                   freq1=self.microwave_frequency,
                                                   phase1=0,
                                                   amp2=rf_amp,
                                                   freq2=radiofreq_2,
                                                   phase2=180)

        pi_y_element_rf2 = self._get_mw_rf_element(length=self.rabi_period / 2,
                                                   increment=0,
                                                   amp1=self.microwave_amplitude,
                                                   freq1=self.microwave_frequency,
                                                   phase1=90.0,
                                                   amp2=rf_amp,
                                                   freq2=radiofreq_2,
                                                   phase2=180)

        pi_minus_x_element_rf2 = self._get_mw_rf_element(length=self.rabi_period / 2,
                                                         increment=0,
                                                         amp1=self.microwave_amplitude,
                                                         freq1=self.microwave_frequency,
                                                         phase1=180.0,
                                                         amp2=rf_amp,
                                                         freq2=radiofreq_2,
                                                         phase2=180)

        pi_minus_y_element_rf2 = self._get_mw_rf_element(length=self.rabi_period / 2,
                                                         increment=0,
                                                         amp1=self.microwave_amplitude,
                                                         freq1=self.microwave_frequency,
                                                         phase1=270.0,
                                                         amp2=rf_amp,
                                                         freq2=radiofreq_2,
                                                         phase2 = 180)

        pi3half_element_rf2 = self._get_mw_rf_gate_element(length=self.rabi_period / 4,
                                                           increment=0,
                                                           amp1=self.microwave_amplitude,
                                                           freq1=self.microwave_frequency,
                                                           phase1=180.0,
                                                           amp2=rf_amp,
                                                           freq2=radiofreq_2,
                                                           phase2=180)

        sequence_list = []
        i = 0

        for index in range(len(tau_array)):

            t = tau_array[index]
            t2 = tau_array2[index]

            tauhalf_element_rf1 = self._get_mw_rf_element(length=t / 2 - self.rabi_period / 4,
                                                 increment=0,
                                                 amp1=0.0,
                                                 freq1=self.microwave_frequency,
                                                 phase1=0,
                                                 amp2=rf_amp,
                                                 freq2=radiofreq_1,
                                                 phase2=0)

            tau_element_rf1 = self._get_mw_rf_element(length=t - self.rabi_period / 2,
                                                 increment=0,
                                                 amp1=0.0,
                                                 freq1=self.microwave_frequency,
                                                 phase1=0,
                                                 amp2=rf_amp,
                                                 freq2=radiofreq_1,
                                                 phase2=0)

            tauhalf_element_rf2 = self._get_mw_rf_element(length=t2 / 2 - self.rabi_period / 4,
                                                       increment=0,
                                                       amp1=0.0,
                                                       freq1=self.microwave_frequency,
                                                       phase1=0,
                                                       amp2=rf_amp,
                                                       freq2=radiofreq_2,
                                                       phase2=180)

            tau_element_rf2 = self._get_mw_rf_element(length=t2 - self.rabi_period / 2,
                                                   increment=0,
                                                   amp1=0.0,
                                                   freq1=self.microwave_frequency,
                                                   phase1=0,
                                                   amp2=rf_amp,
                                                   freq2=radiofreq_2,
                                                   phase2=180)

            wfm_list = []
            wfm_list.extend([pihalf_element_rf1, tauhalf_element_rf1])

            for j in range(xy16_order - 1):
                wfm_list.extend([pi_x_element_rf1, tau_element_rf1, pi_y_element_rf1, tau_element_rf1, pi_x_element_rf1,
                                 tau_element_rf1, pi_y_element_rf1, tauhalf_element_rf1, tauhalf_element_rf2,

                                 pi_y_element_rf2, tau_element_rf2, pi_x_element_rf2, tau_element_rf2, pi_y_element_rf2,
                                 tau_element_rf2, pi_x_element_rf2, tau_element_rf2, pi_minus_x_element_rf2,
                                 tau_element_rf2, pi_minus_y_element_rf2, tau_element_rf2, pi_minus_x_element_rf2,
                                 tau_element_rf2, pi_minus_y_element_rf2, tauhalf_element_rf2, tauhalf_element_rf1,

                                 pi_minus_y_element_rf1, tau_element_rf1, pi_minus_x_element_rf1, tau_element_rf1,
                                 pi_minus_y_element_rf1, tau_element_rf1, pi_minus_x_element_rf1, tau_element_rf1])

            wfm_list.extend([pi_x_element_rf1, tau_element_rf1, pi_y_element_rf1, tau_element_rf1, pi_x_element_rf1,
                             tau_element_rf1, pi_y_element_rf1, tauhalf_element_rf1, tauhalf_element_rf2,

                             pi_y_element_rf2, tau_element_rf2, pi_x_element_rf2, tau_element_rf2, pi_y_element_rf2,
                             tau_element_rf2, pi_x_element_rf2, tau_element_rf2, pi_minus_x_element_rf2,
                             tau_element_rf2, pi_minus_y_element_rf2, tau_element_rf2, pi_minus_x_element_rf2,
                             tau_element_rf2, pi_minus_y_element_rf2, tauhalf_element_rf2, tauhalf_element_rf1,

                             pi_minus_y_element_rf1, tau_element_rf1, pi_minus_x_element_rf1, tau_element_rf1,
                             pi_minus_y_element_rf1, tau_element_rf1, pi_minus_x_element_rf1, tauhalf_element_rf1,
                             last_pihalf_element_rf1])

            wfm_list.extend([laser_element, delay_element, waiting_element])

            name1 = 'LGu_%02i' % i
            blocks, decoupling = self._generate_waveform(wfm_list, name1, np.arange(num_of_points) * tau_step, 1)
            created_blocks.extend(blocks)
            created_ensembles.append(decoupling)

            sequence_list.append((decoupling, {'repetitions': 1, 'trigger_wait': 0, 'go_to': 0, 'event_jump_to': 0}))

            if alternating:
                wfm_list2 = []
                wfm_list2.extend([pihalf_element_rf1, tauhalf_element_rf1])

                for j in range(xy16_order - 1):
                    wfm_list2.extend(
                        [pi_x_element_rf1, tau_element_rf1, pi_y_element_rf1, tau_element_rf1, pi_x_element_rf1,
                         tau_element_rf1, pi_y_element_rf1, tauhalf_element_rf1, tauhalf_element_rf2,

                         pi_y_element_rf2, tau_element_rf2, pi_x_element_rf2, tau_element_rf2, pi_y_element_rf2,
                         tau_element_rf2, pi_x_element_rf2, tau_element_rf2, pi_minus_x_element_rf2,
                         tau_element_rf2, pi_minus_y_element_rf2, tau_element_rf2, pi_minus_x_element_rf2,
                         tau_element_rf2, pi_minus_y_element_rf2, tauhalf_element_rf2, tauhalf_element_rf1,

                         pi_minus_y_element_rf1, tau_element_rf1, pi_minus_x_element_rf1, tau_element_rf1,
                         pi_minus_y_element_rf1, tau_element_rf1, pi_minus_x_element_rf1, tau_element_rf1])

                wfm_list2.extend([pi_x_element_rf1, tau_element_rf1, pi_y_element_rf1, tau_element_rf1, pi_x_element_rf1,
                                 tau_element_rf1, pi_y_element_rf1, tauhalf_element_rf1, tauhalf_element_rf2,

                                 pi_y_element_rf2, tau_element_rf2, pi_x_element_rf2, tau_element_rf2, pi_y_element_rf2,
                                 tau_element_rf2, pi_x_element_rf2, tau_element_rf2, pi_minus_x_element_rf2,
                                 tau_element_rf2, pi_minus_y_element_rf2, tau_element_rf2, pi_minus_x_element_rf2,
                                 tau_element_rf2, pi_minus_y_element_rf2, tauhalf_element_rf2, tauhalf_element_rf1,

                                 pi_minus_y_element_rf1, tau_element_rf1, pi_minus_x_element_rf1, tau_element_rf1,
                                 pi_minus_y_element_rf1, tau_element_rf1, pi_minus_x_element_rf1, tauhalf_element_rf1,
                                 pi3half_element_rf1])

                wfm_list2.extend([laser_element, delay_element, waiting_element])

                name2 = 'XY16d_%02i' % i
                blocks, decoupling2 = self._generate_waveform(wfm_list2, name2, np.arange(num_of_points) * tau_step, 1)
                created_blocks.extend(blocks)
                created_ensembles.append(decoupling2)
                sequence_list.append(
                    (decoupling2, {'repetitions': 1, 'trigger_wait': 0, 'go_to': 0, 'event_jump_to': 0}))

            i = i + 1

        # ---------------------------------------------------------------------------------------------------------------
        created_sequence = PulseSequence(name=name, ensemble_list=sequence_list, rotating_frame=True)

        created_sequence.measurement_information['alternating'] = True
        created_sequence.measurement_information['laser_ignore_list'] = list()
        created_sequence.measurement_information['controlled_variable'] = tau_array
        created_sequence.measurement_information['units'] = ('s', '')
        created_sequence.measurement_information['number_of_lasers'] = num_of_points
        # created_sequence.measurement_information['counting_length'] = self._get_ensemble_count_length(
        #     ensemble=block_ensemble, created_blocks=created_blocks)
        # created_sequence.sampling_information = dict()

        created_sequences.append(created_sequence)

        return created_blocks, created_ensembles, created_sequences