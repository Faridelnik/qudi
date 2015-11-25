# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:31:16 2015

@author: s_ntomek
"""

from socket import socket, AF_INET, SOCK_STREAM
from ftplib import FTP
from io import StringIO
import time
from collections import OrderedDict
from core.base import Base
from core.util.mutex import Mutex
import numpy as np
import hdf5storage
from hardware.pulser_interface import PulserInterface

class AWG70K(Base, PulserInterface):
    """ UNSTABLE: Nikolas
    """
    _modclass = 'awg70k'
    _modtype = 'hardware'
    
    # declare connectors
    _out = {'awg70k': 'PulserInterface'}

    def __init__(self,manager, name, config = {}, **kwargs):

        state_actions = {'onactivate'   : self.activation,
                         'ondeactivate' : self.deactivation}

        Base.__init__(self, manager, name, config, state_actions, **kwargs)
        
        if 'awg_IP_address' in config.keys():
            self.ip_address = config['awg_IP_address']
        else:
            self.logMsg('This is AWG: Did not find >>awg_IP_address<< in '
                        'configuration.', msgType='error')
            
        if 'awg_port' in config.keys():
            self.port = config['awg_port']
        else:
            self.logMsg('This is AWG: Did not find >>awg_port<< in '
                        'configuration.', msgType='error')
        
        self.sample_rate = 25e9
        self.pp_voltage = 0.5
        self.uploaded_sequence_list = []
        self.current_loaded_file = None
        self.is_output_enabled = True
        
        self.use_sequencer = False
        
        self.waveform_directory = '/waves'
        
        self.active_channel = (2,4)
        self.interleave = False

        self.current_status =  0
        
    
    def activation(self, e):
        """ Initialisation performed during activation of the module.
        """
        # connect ethernet socket and FTP        
        self.soc = socket(AF_INET, SOCK_STREAM)
        self.soc.settimeout(3)
        self.soc.connect((self.ip_address, self.port))
        self.ftp = FTP(self.ip_address)
        self.ftp.login()
        self.ftp.cwd('/waves') # hardcoded default folder
        
        self.input_buffer = int(2 * 1024)
        
        self.connected = True
        
        self.update_sequence_list()
        
    
    def deactivation(self, e):
        """Tasks that are required to be performed during deactivation of the module.
        """
        # Closes the connection to the AWG via ftp and the socket
        self.tell('\n')
        self.soc.close()
        self.ftp.close()

        self.connected = False
        pass
    
    def get_constraints(self):
        """ Retrieve the hardware constrains from the Pulsing device.

        @return dict: dict with constraints for the sequence generation and GUI

        Provides all the constraints (e.g. sample_rate, amplitude,
        total_length_bins, channel_config, ...) related to the pulse generator
        hardware to the caller.
        Each constraint is a tuple of the form
            (min_value, max_value, stepsize)
        and the key 'channel_map' indicates all possible combinations in usage
        for this device.

        The possible keys in the constraint are defined here in the interface
        file. If the hardware does not support the values for the constraints,
        then insert just None.
        If you are not sure about the meaning, look in other hardware files
        to get an impression.
        """


        constraints = {}
        # (min, max, incr) in samples/second:
        constraints['sample_rate'] = (1.5e3, 25.0e9, 1)
        # (min, max, res) in Volt-peak-to-peak:
        constraints['amplitude_analog'] = (0.25, 0.5, 0.001)
        # (min, max, res, range_min, range_max)
        # min, max and res are in Volt, range_min and range_max in
        # Volt-peak-to-peak:
        constraints['amplitude_digital'] = (-2.0, 5.4, 0.01, 0.2, 7.4)
        # (min, max, granularity) in samples for one waveform:
        constraints['waveform_length'] = (1, 32400000, 1)
        # (min, max, inc) in number of waveforms in system
        constraints['waveform_number'] = (1, 32000, 1)
        # (min, max, inc) number of subsequences within a sequence:
        constraints['subsequence_number'] = (1, 8000, 1)
        # number of possible elements within a sequence
        constraints['sequence_elements'] = (1, 4000, 1)
        # (min, max, incr) in Samples:
        constraints['total_length_bins'] = (1, 8e9, 1)
        # (analogue, digital) possible combination in given channels:
        constraints['channel_config'] = [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2), (2,3), (2,4)]
        return constraints
    
    def _write_to_file(self, name, ana_samples, digi_samples, sampling_rate, pp_voltage):
        matcontent = {}
        
        matcontent[u'Waveform_Name_1'] = name # each key must be a unicode string
        matcontent[u'Waveform_Data_1'] = ana_samples[0]
        matcontent[u'Waveform_Sampling_Rate_1'] = sampling_rate
        matcontent[u'Waveform_Amplitude_1'] = pp_voltage
        
        if ana_samples.shape[0] == 2:
            matcontent[u'Waveform_Name_1'] = name + '_Ch1'
            matcontent[u'Waveform_Name_2'] = name + '_Ch2'
            matcontent[u'Waveform_Data_2'] = ana_samples[1]
            matcontent[u'Waveform_Sampling_Rate_2'] = sampling_rate
            matcontent[u'Waveform_Amplitude_2'] = pp_voltage
        
        if digi_samples.shape[0] >= 1:
            matcontent[u'Waveform_M1_1'] = digi_samples[0]
        if digi_samples.shape[0] >= 2:
            matcontent[u'Waveform_M2_1'] = digi_samples[1]
        if digi_samples.shape[0] >= 3:
            matcontent[u'Waveform_M1_2'] = digi_samples[2]
        if digi_samples.shape[0] >= 4:
            matcontent[u'Waveform_M2_2'] = digi_samples[3]
        
        hdf5storage.write(matcontent, '.', name+'.mat', matlab_compatible=True)
        return


    def pulser_on(self):
        """ Switches the pulsing device on.

        @return int: error code (0:OK, -1:error)
        """

        self.tell('AWGC:RUN\n')
        self.current_status = 1
        return 0

    def pulser_off(self):
        """ Switches the pulsing device off.

        @return int: error code (0:OK, -1:error)
        """
        self.tell('AWGC:STOP\n')
        self.current_status = 0
        return 0

    def download_waveform(self, waveform, write_to_file = True):
        """ Convert the pre-sampled numpy array to a specific hardware file.

        @param Waveform() waveform: The raw sampled pulse sequence.
        @param bool write_to_file: Flag to indicate if the samples should be
                                   written to a file (= True) or uploaded
                                   directly to the pulse generator channels
                                   (= False).

        @return int: error code (0:OK, -1:error)

        Brings the numpy arrays containing the samples in the Waveform() object
        into a format the hardware understands. Optionally this is then saved
        in a file. Afterwards they get downloaded to the Hardware.
        """

        if write_to_file:
            self._write_to_file(waveform.name, waveform.analogue_samples, waveform.digital_samples, waveform.sampling_freq, waveform.pp_voltage)
            
            # TODO: Download waveform to AWG and load it into channels
            self.send_file(self.waveform_directory + waveform.name + '.mat')
            self.load_sequence(waveform.name)
        return 0

    def send_file(self, filepath):
        """ Sends an already hardware specific waveform file to the pulse
            generators waveform directory.

        @param string filepath: The file path of the source file

        @return int: error code (0:OK, -1:error)

        Unused for digital pulse generators without sequence storage capability
        (PulseBlaster, FPGA).
        """
        self.uploaded_sequence_list.append(filepath.rsplit('/', 1)[1][:-4])
        return 0
    
    def load_sequence(self, seq_name, channel = None):
        """ Loads a sequence to the specified channel of the pulsing device.

        @param str seq_name: The name of the sequence to be loaded
        @param int channel: The channel for the sequence to be loaded into if
                            not already specified in the sequence itself

        @return int: error code (0:OK, -1:error)

        Unused for digital pulse generators without sequence storage capability
        (PulseBlaster, FPGA).
        """
        if seq_name in self.uploaded_sequence_list:
            self.clear_all()
            file_path  = 'C:/inetpub/ftproot' + self.waveform_directory + '/' + seq_name + '.mat'
            self.tell('MMEM:OPEN:SASS:WAV "%s"\n' % file_path)
            self.tell('*OPC?\n')
            wfm_num = int(self.ask('WLISt:SIZE?\n'))
            if wfm_num == 1:
                wfm_name = seq_name + '_Ch1'
                self.tell('SOUR1:CASS:WAV "%s"\n' % wfm_name)
            elif wfm_num == 2:
                wfm_name = seq_name + '_Ch1'
                self.tell('SOUR1:CASS:WAV "%s"\n' % wfm_name)
                wfm_name = seq_name + '_Ch2'
                self.tell('SOUR2:CASS:WAV "%s"\n' % wfm_name)
            self.current_loaded_file = seq_name
        return 0
                
    def clear_all(self):
        """ Clears all loaded waveform from the pulse generators RAM
        Unused for digital pulse generators without storage capability (PulseBlaster, FPGA).
        
        @return int: error code (0:OK, -1:error)
        """
        self.tell('WLIS:WAV:DEL ALL\n')
        self.current_loaded_file = None
        return 0
        
    def get_status(self):
        """ Retrieves the status of the pulsing hardware

        @return (int, dict): inter value of the current status with the
                             corresponding dictionary containing status
                             description for all the possible status variables
                             of the pulse generator hardware
        """
        status_dic = {}
        status_dic[-1] = 'Failed Request or Communication'
        status_dic[0] = 'Device has stopped, but can receive commands.'
        status_dic[1] = 'Device is active and running.'
        # All the other status messages should have higher integer values
        # then 1.

        return (self.current_status, status_dic)
    
    def set_sample_rate(self, sample_rate):
        """ Set the sample rate of the pulse generator hardware
        
        @param float sample_rate: The sample rate to be set (in Hz)
        
        @return foat: the sample rate returned from the device (-1:error)
        """
        self.tell('CLOCK:SRATE %.4G\n' % sample_rate)
        time.sleep(3)
        return_rate = float(self.ask('CLOCK:SRATE?\n'))
        self.sample_rate = return_rate
        return return_rate
        
    def get_sample_rate(self):
        """ Set the sample rate of the pulse generator hardware
        
        @return float: The current sample rate of the device (in Hz)
        """
        return self.sample_rate
        
    def set_pp_voltage(self, channel, voltage):
        """ Set the peak-to-peak voltage of the pulse generator hardware analogue channels.
        Unused for purely digital hardware without logic level setting capability (DTG, FPGA, etc.).

        @param int channel: The channel to be reconfigured
        @param float amplitude: The peak-to-peak amplitude the channel should be set to (in V)

        @return int: error code (0:OK, -1:error)
        """
        self.pp_voltage = voltage
        return 0

    def get_pp_voltage(self, channel):
        """ Get the peak-to-peak voltage of the pulse generator hardware analogue channels.

        @param int channel: The channel to be checked

        @return float: The peak-to-peak amplitude the channel is set to (in V)

        Unused for purely digital hardware without logic level setting
        capability (FPGA, etc.).
        """
        return self.pp_voltage

    def set_active_channels(self, d_ch=0, a_ch=0):
        """ Set the active channels for the pulse generator hardware.

        @param int d_ch: Number of digital channels
        @param int a_ch: Number of analogue channels

        @return int: error code (0:OK, -1:error)
        """
        # FIXME: That is not a good way of setting the active channels since no
        # deactivation method of the channels is provided.
        if d_ch <= 2:
            ch1_marker = d_ch
            ch2_marker = 0
        elif d_ch == 3:
            ch1_marker = 2
            ch2_marker = 1
        else:
            ch1_marker = 2
            ch2_marker = 2

        self.tell('SOURCE1:DAC:RESOLUTION ' + str(10-ch1_marker) + '\n')
        self.tell('SOURCE2:DAC:RESOLUTION ' + str(10-ch2_marker) + '\n')


        if a_ch == 2:
            self.tell('OUTPUT2:STATE ON\n')
            self.tell('OUTPUT1:STATE ON\n')
        elif a_ch ==1:
            self.tell('OUTPUT1:STATE ON\n')
            self.tell('OUTPUT2:STATE OFF\n')
        else:
            self.tell('OUTPUT1:STATE OFF\n')
            self.tell('OUTPUT2:STATE OFF\n')
            
        self.active_channel = (a_ch, d_ch)
        return 0

    def get_active_channels(self):
        """ Get the active channels of the pulse generator hardware.

        @return (int, int): number of active channels (analogue, digital)
        """
        return self.active_channel

    def get_sequence_names(self):
        """ Retrieve the names of all downloaded sequences on the device.

        @return list: List of sequence name strings

        Unused for digital pulse generators without sequence storage capability
        (PulseBlaster, FPGA).
        """

        with FTP(self.ip_address) as ftp:
            ftp.login() # login as default user anonymous, passwd anonymous@
            ftp.cwd(self.waveform_directory)

            # get only the files from the dir and skip possible directories
            log =[]
            file_list = []
            ftp.retrlines('LIST', callback=log.append)
            for line in log:
                if not '<DIR>' in line:
                    file_list.append(line.rsplit(None, 1)[1][:-4])
        return file_list
        
    def update_sequence_list(self):
        """
        Updates uploaded_sequence_list 
        """
        self.uploaded_sequence_list = self.get_sequence_names()
        return 0
        

    def delete_sequence(self, seq_name):
        """ Delete a sequence with the passed seq_name from the device memory.

        @param str seq_name: The full name of the file to be deleted.
                             Optionally a list of file names can be passed.

        @return int: error code (0:OK, -1:error)

        Unused for digital pulse generators without sequence storage capability
        (PulseBlaster, FPGA).
        """
        if not isinstance(seq_name, list):
            seq_name = [seq_name]

        with FTP(self.ip_address) as ftp:
            ftp.login() # login as default user anonymous, passwd anonymous@
            ftp.cwd(self.waveform_directory)

            for entry in seq_name:
                if entry in self.uploaded_sequence_list:
                    ftp.delete(entry + '.mat')
                    self.uploaded_sequence_list.remove(entry)
                    if entry == self.current_loaded_file:
                        self.clear_all()        
        return 0
        
        
    def set_sequence_directory(self, dir_path):
        """ Change the directory where the sequences are stored on the device.
        
        @param string dir_path: The target directory
        
        @return int: error code (0:OK, -1:error)

        Unused for digital pulse generators without sequence storage
        capability (PulseBlaster, FPGA).
        """

        # check whether the desired directory exists:
        with FTP(self.ip_address) as ftp:
            ftp.login() # login as default user anonymous, passwd anonymous@

            try:
                ftp.cwd(dir_path)
            except:
                self.logMsg('Desired directory {0} not found on AWG device.\n'
                            'Create new.'.format(dir_path), msgType='status')
                ftp.mkd(dir_path)

        self.waveform_directory = dir_path
        return 0
       
    def get_sequence_directory(self):
        """ Ask for the directory where the sequences are stored on the device.
        
        @return string: The current sequence directory

        Unused for digital pulse generators without sequence storage capability
        (PulseBlaster, FPGA).
        """
        return self.waveform_directory

    def set_interleave(self, state=False):
        # TODO: Implement this function
        """ Turns the interleave of an AWG on or off.

        @param bool state: The state the interleave should be set to
                           (True: ON, False: OFF)

        @return int: error code (0:OK, -1:error)

        Unused for pulse generator hardware other than an AWG.
        """
        return 0

    def get_interleave(self):
        # TODO: Implement this function
        """ Check whether Interleave is on in AWG.

        @return bool: True: ON, False: OFF

        Unused for pulse generator hardware other than an AWG.
        """
        return self.interleave

    def tell(self, command):
        """Send a command string to the AWG.
        
        @param command: string containing the command
        
        @return int: error code (0:OK, -1:error)
        """
        if not command.endswith('\n'): 
            command += '\n'
        command = bytes(command, 'UTF-8') # In Python 3.x the socket send command only accepts byte type arrays and no str
        self.soc.send(command)
        return 0
        
    def ask(self, question):
        """Asks the device a 'question' and receive and return an answer from device.
        
        @param string question: string containing the command
        
        @return string: the answer of the device to the 'question'
        """
        if not question.endswith('\n'): 
            question += '\n'
        question = bytes(question, 'UTF-8') # In Python 3.x the socket send command only accepts byte type arrays and no str
        self.soc.send(question)    
        time.sleep(0.1)                 # you need to wait until AWG generating
                                        # an answer.
        message = self.soc.recv(self.input_buffer)  # receive an answer
        message = message.decode('UTF-8') # decode bytes into a python str
        message = message.replace('\n','')      # cut away the characters\r and \n.
        message = message.replace('\r','')
        return message
        
    def reset(self):
        """Reset the device.
        
        @return int: error code (0:OK, -1:error)
        """
        self.tell('*RST\n')
        return 0
        
    

#TODO: ------------------------------------------------------------------------- has to be reworked -----------------------------------------
        
#    def get_status(self):
#        """ Asks the current state of the AWG.
#        @return: an integer with the following meaning: 
#                0 indicates that the instrument has stopped.
#                1 indicates that the instrument is waiting for trigger.
#                2 indicates that the instrument is running.
#               -1 indicates that the request of the status for AWG has failed.
#                """
#        self.soc.connect((self.ip_address, self.port))
#        self.soc.send('AWGC:RSTate?\n') # send at first a command to request.
#        time.sleep(1)                   # you need to wait until AWG generating
#                                        # an answer.
#        message = self.soc.recv(self.input_buffer)  # receive an answer
#        self.soc.close()
#        # the output message contains always the string '\r\n' at the end. Use
#        # the split command to get rid of this
#        try:
#            return int(message.split('\r\n',1)[0])
#        except:
#            # if nothing comes back than the output should be marked as error
#            return -1
            
#    def get_sequencer_mode(self, output_as_int=False):
#        """ Asks the AWG which sequencer mode it is using. It can be either in 
#        Hardware Mode or in Software Mode. The optional variable output_as_int
#        sets if the returned value should be either an integer number or string.
#        
#        @param: output_as_int: optional boolean variable to set the output
#        @return: an string or integer with the following meaning:
#                'HARD' or 0 indicates Hardware Mode
#                'SOFT' or 1 indicates Software Mode
#                'Error' or -1 indicates a failure of request
#        """
#        self.soc.connect((self.ip_address, self.port))
#        self.soc.send('AWGControl:SEQuencer:TYPE?\n')
#        time.sleep(1)
#        message = self.soc.recv(self.input_buffer)
#        self.soc.close()
#        if output_as_int == True:
#            if 'HARD' in message:
#                return 0
#            elif 'SOFT' in message:
#                return 1
#            else:
#                return -1
#        else:
#            if 'HARD' in message:
#                return 'Hardware-Sequencer'
#            elif 'SOFT' in message:
#                return 'Software-Sequencer'
#            else:
#                return 'Request-Error'
#                
#    def set_Interleave(self, state=False):
#        """Turns Interleave of the AWG on or off.
#            @param state: A Boolean, defines if Interleave is turned on or off, Default=False
#        """
#        self.soc.connect((self.ip_address, self.port))
#        if(state):
#            print('interleave is on')
#            self.soc.send('AWGC:INT:STAT 1\n')
#        else:
#            print('interleave is off')
#            self.soc.send('AWGC:INT:STAT 0\n')
#        self.soc.close()
#        return    
    
#    def set_output(self, state, channel=3):
#        """Set the output state of specified channels.
#        
#        @param state:  on : 'on', 1 or True; off : 'off', 0 or False
#        @param channel: integer,   1 : channel 1; 2 : channel 2; 3 : both (default)
#        
#        """
#        #TODO: AWG.set_output: implement swap
#        look_up = {'on' : 1, 1 : 1, True : 1,
#                   'off' : 0, 0 : 0, False : 0
#                  }
#        self.soc.connect((self.ip_address, self.port))
#        if channel & 1 == 1:
#            self.soc.send('OUTP1 %i\n' % look_up[state])
#        if channel & 2 == 2:
#            self.soc.send('OUTP2 %i\n' % look_up[state])
#        self.soc.close()
#        return
#        
#    def set_mode(self, mode):
#        """Change the output mode.
#
#        @param  mode: Options for mode (case-insensitive):
#        continuous - 'C'
#        triggered  - 'T'
#        gated      - 'G'
#        sequence   - 'S'
#        
#        """
#        look_up = {'C' : 'CONT',
#                   'T' : 'TRIG',
#                   'G' : 'GAT' ,
#                   'E' : 'ENH' , 
#                   'S' : 'SEQ'
#                  }
#        self.soc.connect((self.ip_address, self.port))
#        self.soc.send('AWGC:RMOD %s\n' % look_up[mode.upper()])
#        self.soc.close()
#        return
#        
#    def set_sample(self, frequency):
#        """Set the output sampling rate.
#        
#        @param frequency: sampling rate [GHz] - min 5.0E-05 GHz, max 24.0 GHz 
#        """
#        self.soc.connect((self.ip_address, self.port))
#        self.soc.send('SOUR:FREQ %.4GGHz\n' % frequency)
#        self.soc.close()
#        return
#        
#    def set_amp(self, voltage, channel=3):
#        """Set output peak-to-peak voltage of specified channel.
#        
#        @param voltage: output Vpp [V] - min 0.05 V, max 2.0 V, step 0.001 V
#        @param channel:  1 : channel 1; 2 : channel 2; 3 : both (default)
#        
#        """
#        self.soc.connect((self.ip_address, self.port))
#        if channel & 1 == 1:
#            self.soc.send('SOUR1:VOLT %.4GV\n' % voltage)
#        if channel & 2 == 2:
#            self.soc.send('SOUR2:VOLT %.4GV\n' % voltage)
#        self.soc.close()
#        return
#    
#    def set_jump_timing(self, synchronous = False):
#        """Sets control of the jump timing in the AWG to synchoronous or asynchronous.
#        If the Jump timing is set to asynchornous the jump occurs as quickly as possible 
#        after an event occurs (e.g. event jump tigger), if set to synchornous 
#        the jump is made after the current waveform is output. The default value is asynchornous
#        
#        @param synchronous: Bool, if True the jump timing will be set to synchornous, 
#        if False the jump timing will be set to asynchronous
#        """
#        self.soc.connect((self.ip_address, self.port))
#        if(synchronous):
#            self.soc.send('EVEN:JTIM SYNC\n')
#        else:
#            self.soc.send('EVEN:JTIM ASYNC\n')
#        self.soc.close()
#        return
#            
#    def load(self, filename, channel=1, cwd=None):
#        """Load sequence or waveform file into RAM, preparing it for output.
#        
#        Waveforms and single channel sequences can be assigned to each or both
#        channels. Double channel sequences must be assigned to channel 1.
#        The AWG's file system is case-sensitive.
#        
#        @param filename:  *.SEQ or *.WFM file name in AWG's CWD
#        @param channel: 1 : channel 1 (default); 2 : channel 2; 3 : both
#        @param cwd: filepath where the waveform to be loaded is stored. Default: 'C:\InetPub\ftproot\waves'
#        """
#        if cwd is None:
#            cwd = 'C:\\InetPub\\ftproot\\waves' # default
#        self.soc.connect((self.ip_address, self.port))
#        if channel & 1 == 1:
#            self.soc.send('SOUR1:FUNC:USER "%s/%s"\n' % (cwd, filename))
#        if channel & 2 == 2:
#            self.soc.send('SOUR2:FUNC:USER "%s/%s"\n' % (cwd, filename))
#        self.soc.close()
#        return
#    
#    def clear_AWG(self):
#        """ Delete all waveforms and sequences from Hardware memory and clear the visual display """
#        self.soc.connect((self.ip_address, self.port))
#        self.soc.send('WLIS:WAV:DEL ALL\n')
#        self.soc.close()
#        return
#    
#
#    def max_samplerate(self):
#        return_val = self.max_samplerate
#        return return_val