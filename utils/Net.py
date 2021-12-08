from neuron import h
from neuron.units import mV, ms
import numpy as np
import math
import sys
h.load_file('stdrun.hoc')

import matplotlib.pyplot as plt


sys.path.insert(1, "./")
from Cell import HH, INF


class Net:
    def __init__(self, N, excitatory_ratio, Cell):
        # Cell: the Cell class (HH or INF) used in this model
        self.Cell = Cell
        self.cells = []
        self.netcons = [] # store the netcons

        self.edge_mat = np.zeros((N, N)) # store cell connections here: (rows = source, cols = target)
        np.fill_diagonal(self.edge_mat, 1)

        self.cell_locations = np.random.rand(N, 2) # generate x and y locations for cells
        
        # cell categories go as follows:
            # -1: inhibatory
            # +1: excitatory
        self.cell_categories = np.random.choice([-1,1], N, p=[1-excitatory_ratio, excitatory_ratio])
        
        for i in range(N):        
            _cell = Cell(polarity=self.cell_categories[i], x=self.cell_locations[i, 0], y=self.cell_locations[i, 1])
            self.cells.append(_cell)
            
        self.average_cell_dist = 0.5214054 # This is the average distance between 2 points random points in a unit square
            
    def connect_cell_pair(self, source, target, delay, weight):
        # source, target: ints for the indices of the 2 respective cells
        source_cell = self.cells[source]
        target_cell = self.cells[target]

        if self.Cell == HH:
            nc = h.NetCon(source_cell.axon(0.5)._ref_v, target_cell.syns[source_cell.polarity]['syn'], sec=source_cell.axon)
            nc.delay = delay
            nc.weight[0] = weight
        elif self.Cell == INF:
            nc = h.NetCon(source_cell.cell, target_cell.cell)
            nc.delay = delay
            if source_cell.polarity == 1:
                nc.weight[0] = source_cell.polarity * weight / 4
            else:
                nc.weight[0] = source_cell.polarity * weight
        else:
            print('error in cell connection')
        
        self.netcons.append({'source':source, 'target':target, 'netcon':nc})

    def randomly_connect_cells(self, delay, weight):
        # delay:
            # can be a float: (ms)
            # scaled by distance between cells if weight_scaling == True
        degrees = [2,3,4,5,6,7]
        degrees_p = [0.15, 0.2, 0.3, 0.2, 0.1, 0.05]
        
        for source in range(len(self.cells)):
            # remove this cell and cells that point to this cell from potential targets
            potential_targets = [target for target in range(len(self.cells)) if target != source and self.edge_mat[source, target] == 0]
            node_degree = min(len(potential_targets), np.random.choice(degrees, 1, degrees_p))
            targets = np.random.choice(potential_targets, node_degree, replace=False)
            for target in targets:
                self.edge_mat[source, target] = 1
                self.connect_cell_pair(source, target, delay, weight)
                
    def initial_stimulation(self, n_cells, n_stimuli=1, delay=0, weight=.2, stim_interval=10):
        # n_cells: number of cells to stimulate
        # n_stimuli: number of stimuli to add to each cell
        
        # choose n_cells excitatory cells to stimulate
        excitatory_inds = np.argwhere(self.cell_categories == 1).reshape(-1)
        self.inds2recieve_poisson_stim = np.random.choice(excitatory_inds, min(n_cells, len(excitatory_inds)), replace=False)
        for cell_ind in self.inds2recieve_poisson_stim:
            self.cells[cell_ind].add_poisson_stimulus('initial_poisson', 1, stim_interval, delay, weight, n_stimuli=n_stimuli)

    def aggregate_all_spikes(self):
        # aggregate all spikes
        self.ex_all_spikes = []
        self.ex_num_cells = 0
        self.in_all_spikes = []
        self.in_num_cells = 0    
        for cell in self.cells:
            if cell.polarity == 1: # excitatory
                self.ex_all_spikes.extend(list(cell.spike_times))
                self.ex_num_cells += 1
            elif cell.polarity == -1: # inhibitory
                self.in_all_spikes.extend(list(cell.spike_times))
                self.in_num_cells += 1
            else:
                print('error in cell category during spike aggregation')

    def show_cell_grid(self):
        plt.figure(figsize = (10,10))
        
        # plot cells
        plt.scatter(
            self.cell_locations[:,0], 
            self.cell_locations[:,1], 
            c=self.cell_categories, 
            cmap='bwr',
            marker = 'o',
            s = 100,
            alpha = .5
        )
        # plot netcons
        for nc in self.netcons:
            source = [self.cells[nc['source']].x, self.cells[nc['source']].y]
            target = [self.cells[nc['target']].x, self.cells[nc['target']].y]
            
            plt.arrow(
                x = source[0], y = source[1], 
                dx = target[0] - source[0], 
                dy = target[1] - source[1],
                head_width = .01,
                length_includes_head = True,
                alpha = .5
            )
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.grid()
        plt.show()
