import numpy
from .autodiff import Layer

class SubsamplingLayer(Layer):
    """
    $F(v, W) = P_U v$ with projector $P_U$ on sub-space $U$.
    Note that the sub-space projection only works on a cartesian sub-space.
    The projector is given as a numpy array, where the elements of the array are 
    a list of elements to pick.
    """
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

    def get_dot_converter(self, cid):
        dot_node_name = self.dot_get_node_name(cid)
        dot_node_descr = f"{dot_node_name} [label=\"SubSample {len(self._sample)}\"]"
        return (dot_node_name, dot_node_name, [dot_node_descr], [])

