import numpy as np
import logging
logger = logging.getLogger(__name__)

minval = 1e-100

class Network(object):
    def __init__(self, num_output_neurons, num_input_neurons, max_entry, counts=None):
        self.num_output_neurons = num_output_neurons
        self.num_input_neurons = num_input_neurons
        self.max_entry = max_entry

        if counts is None:
            logger.debug("Initializing network with fresh counts.")
            self.counts = self.make_counts()
            self.normalize_counts()
        else:
            logger.debug("Initializing network with provided counts.")
            self.counts = counts
            expected_shape = (num_output_neurons, num_input_neurons, max_entry + 1)
            if self.counts.shape != expected_shape:
                raise ValueError(
                    "Supplied counts do not have the expected shape ((%d,%d,%d) != (%d,%d,%d))." %
                    (self.counts.shape + expected_shape)
                )

        self.gridsel = np.ogrid[:num_output_neurons,:num_input_neurons]

    def make_counts(self):
        counts = np.random.normal(
            loc  = minval, scale=minval,
            size = (self.num_output_neurons, self.num_input_neurons, self.max_entry + 1))
        return counts.clip(minval / 10,1).astype(np.float128)

    def map_one(self, data):
        probs = self.counts[self.gridsel[0], self.gridsel[1], data]
        probs = probs.prod(axis=1)
        return probs.argmax()

    def map_many(self, data):
        return np.array([self.map_one(d) for d in data])
    
    def update(self, data, sigma, alpha, bmus = None):
        if bmus is None:
            bmus = self.map_many(data)

        upd = self.make_counts()
        for o in range(self.num_output_neurons):
            relevant = data[bmus == o]
            for dp in relevant:
                for i in range(self.num_input_neurons):
                    upd[o][i][dp[i]] +=  1
        
        filtered = self.interact(upd, sigma)
        self.normalize(filtered)

        if alpha == 1:
            self.counts = filtered
        else:
            self.counts = self.counts * (1-alpha) + filtered * alpha
            self.normalize_counts()

    def normalize(self, what):
        for o in range(self.num_output_neurons):
            for i in range(self.num_input_neurons):
                what[o][i] /= what[o][i].sum()

    def interact(self, upd, sigma):
        # this is probably very expensive.
        # Don't know how to do this more efficiently in np.
        filtered = np.zeros_like(upd)
        upd = np.roll(upd, -upd.shape[0]/2, axis=0)
        for i in range(-upd.shape[0] / 2, upd.shape[0] / 2-1):
            upd = np.roll(upd, 1, axis=0)
            factor = np.exp(-(float(i)/upd.shape[0])**2/(2*sigma**2))
            filtered[:] += factor * upd
        return filtered
        
    def normalize_counts(self):
        self.normalize(self.counts)

    def drop_input_channels(self, dropped):
        undropped = ~ np.asarray(dropped, dtype=bool)
        num_input_neurons = undropped.sum()
        counts = self.counts[:,undropped,:]

        copy = Network(self.num_output_neurons, num_input_neurons, self.max_entry, counts)
    
        return copy
