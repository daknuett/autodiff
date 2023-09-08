import numpy
from .autodiff import Layer

class SubsamplingLayer(Layer):
    def __init__(self, sample):
        super().__init__()
        self._sample = numpy.array(sample)

    def random_weights(self):
        yield numpy.array([0])
    
    def dweights_proj(self, W, vin, r):
        yield numpy.array([0])


    @property
    def nout(self):
        return self._sample.shape[0]

    def to_json(self):
        return {"__type__": self.__class__.__name__
                , "sample": [s for s in self._sample]}

    def call(self, W, vin):
        return vin[self._sample]

    def dvin_proj(self, W, vin, r):
        return numpy.copy(r)

    def generate_and_set_weightlist_dispatcher(self, cindex):
        self._dispatcher = next(cindex)
        yield self._dispatcher
