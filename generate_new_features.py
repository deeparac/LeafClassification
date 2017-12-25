import mxnet as mx
import numpy as np
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def generate_new_features(leafnet):
    sym, arg_params, aux_params = mx.model.load_checkpoint("../leafnet/chkpt", 200)
    nmr_array, img_array, img_label = leafnet.load_train_data()
    activation_features = leafnet.out.get_internals()['fc1_output']
    new_features = []

    viz_mod = mx.mod.Module(symbol=activation_features, 
                             context=mx.gpu(), 
                             label_names=None, 
                             data_names=['img_input', 'numerical']
    )
    viz_mod.bind(for_training=False, 
                  data_shapes=[('img_input', (1, 1, 96, 96)), ('numerical', (1, 192))]
    )
    
    viz_mod.set_params(arg_params=arg_params, aux_params=aux_params)
    for ind, val in enumerate(img_array):
        viz_mod.forward(Batch([mx.nd.array(img_array[ind].reshape(1, 1, 96, 96)),
                               mx.nd.array(nmr_array[ind].reshape(1, 192))]))
        features = viz_mod.get_outputs()[0].asnumpy()
        nonzero_features = []
        for f in features[0]:
            if f != 0.0:
                nonzero_features.append(f)
        new_features.append(np.array(nonzero_features))
    
    return new_features, viz_mod