import copy
from torch.nn import Module
from torch import Tensor


class EMA:
    def __init__(self, beta: float):
        """
        Exponential moving average class.  Exponential moving average gives better results
        for generative models.
        Args :
          beta : decay factor for older data in the average.
        """
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model: Module, current_model: Module):
        """
        Args :
          ma_model : model average so far
          current_model : new model to add to the average
        """
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: Tensor, new: Tensor) -> Module:
        """
        Compute the weighted average of the old weights and the new weights
        Args :
          old : Old weights
          new : New weights
        Returns :
          weighted average
        """
        if old is None:
            # TODO: I think this might be a bug as pretty sure we want a deepcopy here.
            return new  # copy.deepcopy(new)

        return old * self.beta + (1 - self.beta) * new
