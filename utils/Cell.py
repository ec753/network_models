from neuron import h
from neuron.units import mV, ms
import numpy as np
import math
h.load_file('stdrun.hoc')
import matplotlib.pyplot as plt

class HH:
    def __init__(self, polarity, x = 'na', y = 'na'):
        self.polarity = polarity # int polarity refers to 1: excitatory or -1: inhibitory
        self.x = x
        self.y = y

        self.axon = h.Section(name='axon')
        self.axon.insert(h.hh)
        
        # set up synapses
        self.syns = {} # dictionary with {'syn_title':{'syn':h.ExpSyn(), 'current':h.Vector()}}
        # add an excitatory and inhibitory synapse
        self.create_synapse(title=1, tau=2, reversal_potential=0) # excitatory
        self.create_synapse(title=-1, tau=6, reversal_potential=-80) # inhibitory

        # setup stimuli and netcons
        self.stims = {} # dictionary with {'stim_title':, 'syn_title':(-1 or 1), 'stim':h.NetStim(), 'sender':, 'stim_times': h.Vector()}}
        self.netcons = [] 
        
        # setup recording
        self.spike_detector = h.NetCon(self.axon(0.5)._ref_v, None, sec=self.axon)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)
        
        self._t = h.Vector().record(h._ref_t, sec=self.axon)
        self._v = h.Vector().record(self.axon(0.5)._ref_v)
        self._m = h.Vector().record(self.axon(0.5).hh._ref_m)
        self._n = h.Vector().record(self.axon(0.5).hh._ref_n)
        self._h = h.Vector().record(self.axon(0.5).hh._ref_h)

    def create_synapse(self, title, tau, reversal_potential):        
        # add synapse
        syn = h.ExpSyn(self.axon(0))
        syn.tau = tau
        syn.e = reversal_potential * mV
        syn_current = h.Vector().record(syn._ref_i)
        
        self.syns[title] = {'syn':syn, 'current':syn_current}
    
    def add_poisson_stimulus(self, stim_title, syn_title, stim_interval, delay, weight, n_stimuli = 999999):
        # syn_title: refers to the title of the synapse in the syns dictionary (ex: syns['excitatory'])
        
        # add stimulus
        stim = h.NetStim()
        stim.number = n_stimuli
        stim.interval = stim_interval * ms
        stim.noise = True
        stim.start = 0 * ms
        stim_times = h.Vector()
        
        nc = h.NetCon(stim, self.syns[syn_title]['syn'])
        nc.delay = delay * ms
        nc.weight[0] = weight
        nc.record(stim_times)

        # store in the class
        self.stims[stim_title] = {
            'syn_title': syn_title,
            'stim': stim,
            'sender': 'poisson',
            'stim_times': stim_times
        }
        self.netcons.append(nc)
        
    def plot_cell(self):
        fig, axes = plt.subplots(3,1, figsize = (15,10), sharex = True)
        # inputs
        axes[0].set_ylabel('stimuli')
        stim_titles = []
        for i, stim_title in enumerate(self.stims):
            stim_titles.append(stim_title)
            axes[0].vlines(list(self.stims[stim_title]['stim_times']), i, i+1)
        axes[0].set_yticks([i+.5 for i in range(len(self.stims))])
        axes[0].set_yticklabels(stim_titles)
        
        # membrane voltage
        axes[1].set_ylabel('Membrane\nVoltage (mV)')
        axes[1].plot(self._t, self._v)
        
        # HH parameters
        axes[2].set_ylabel('HH parameters')
        axes[2].plot(self._t, self._m)
        axes[2].plot(self._t, self._n)
        axes[2].plot(self._t, self._h)
        
        # cosmetics
        axes[2].set_xlabel('time (ms)')


class INF:
    def __init__(self, polarity, taue=2, taui1=.1, taui2=6, taum=10, x='na', y='na'):
        self.polarity = polarity  # int polarity refers to 1: excitatory or -1: inhibitory
        self.x = x
        self.y = y

        # create cell
        self.cell = h.IntFire4()

        self.cell.taue = taue    # ms excitatory input time constant
        self.cell.taui1 = taui1  # ms inhibitory input rise time constant
        self.cell.taui2 = taui2  # ms inhibitory input fall time constant
        self.cell.taum = taum    # membrane time constant

        # setup stimuli and netcons
        self.stims = {}  # dictionary with {'stim_title': {'syn_title':(-1 or 1), 'stim':h.NetStim(), 'sender':, 'stim_times': h.Vector()} }
        self.netcons = []

        # setup recording
        self.spike_detector = h.NetCon(self.cell, None)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)

    def check_INF_taus(self):
        # check the time constants to see if they meet the needed constraint:
        # taue < taui1 < taui2 < taum
        return self.cell.taue < self.cell.taui1 < self.cell.taui2 < self.cell.taum

    def add_poisson_stimulus(self, stim_title, syn_title, stim_interval, delay, weight, n_stimuli = 999999):
        # add stimulus
        stim = h.NetStim()
        stim.number = n_stimuli
        stim.interval = stim_interval * ms
        stim.noise = True
        stim.start = 0 * ms
        stim_times = h.Vector()

        # connect stimulus to cell
        nc = h.NetCon(stim, self.cell)
        nc.delay = delay * ms
        nc.weight[0] = syn_title * weight # syn_title is 1:excitatory or -1:inhibitory, and h.IntFire4() interprets positive weights as excitatory and negative as inhibitory
        nc.record(stim_times)

        # store in the class
        self.stims[stim_title] = {
            'syn_title': syn_title,
            'stim': stim,
            'sender': 'poisson',
            'stim_times': stim_times
        }
        self.netcons.append(nc)

    def plot_cell(self, sim_length):
        fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

        # inputs
        axes[0].set_ylabel('stimuli')
        stim_titles = []
        for i, stim in enumerate(self.stims):
            stim_titles.append(stim)
            axes[0].vlines(list(self.stims[stim]['stim_times']), i, i + 1)
        axes[0].set_yticks([i + .5 for i in range(len(self.stims))])
        axes[0].set_yticklabels(stim_titles)

        # model output
        axes[1].set_ylabel('output')
        axes[1].vlines(list(self.spike_times), 0, 1)

        axes[0].set_xlim(0, sim_length)
        return

