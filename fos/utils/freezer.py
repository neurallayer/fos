
class Freezer():
    '''Provides functionality to freeze/unfreeze parameters in a model based
    on their name. This comes in handy during transfer learning at
    the beginning of the training when you only want to train the newly added layers.
       
    If you want to know the names of the paramters, use `freezer.summary()`.

    Args:
        model (nn.Module): the model you want to use.

    Example usage:
       
    .. code-block:: python

        freezer = Freezer(my_model)
        freezer.freeze() # freeze all layers
        freezer.unfreeze("fc") # unfreeze layer who's name starts with "fc"
        trainer.run(data, epochs=1)
        freezer.unfreeze() # unfreeze all layers
        trainer.run(data, epochs=50)
    '''

    def __init__(self, model):
        self.model = model

    def _get_params(self, layer_name=""):
        for name, param in self.model.named_parameters():
            if name.startswith(layer_name):
                yield param

    def _set_requires_grad(self, req_grad, layer_name=""):
        for param in self._get_params(layer_name):
            param.requires_grad = req_grad

    def freeze(self, layer_name=""):
        '''Freeze a number of layers based on their name. If no name is provided, it will freeze
           all layers.

           Args:
              layer_name (str): The first part of the layer_name. Can be a single string or a set of strings.
        '''

        self._set_requires_grad(False, layer_name)

    def unfreeze(self, layer_name=""):
        '''Unfreeze a number of layers based on their name. If no name is provided, it will unfreeze
           all layers.

           Args:
              layer_name (str): The first part of the layer_name. Can be a single string or a set of strings.
        '''

        self._set_requires_grad(True, layer_name)

    def summary(self):
        '''Print an overview of the parameters and their status.
        '''
        for idx, (name, layer) in enumerate(self.model.named_parameters()):
            text = "[unfrozen]" if layer.requires_grad else "[frozen]"
            print("{:3} {:10} {:50} {}".format(
                idx, text, name, tuple(layer.shape)))
