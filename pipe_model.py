import torch
import torch.nn as nn


class PipeModel(nn.Module):
    """This class will connect two models together as following. First model perform prediction, if it predicts the last class, the second model will make the final prediction."""

    def __init__(
        self, mcunet: nn.Module, convnext: nn.Module, num_dominnat_classes: int, device
    ):
        """
        Initialize the PipeModel with two models.

        Args:
            mcunet: The first model (e.g., a MobileNetV2 or similar).
            num_dominnat_classes: Number of classes for the first model.
            convnext: The second model (e.g., ConvNeXt).
        """
        super(PipeModel, self).__init__()
        self.model1 = mcunet
        self.num_classes = num_dominnat_classes
        self.model2 = convnext
        self.device = device

    def forward(self, x):
        """Forward pass through the first model and then the second model if needed."""
        # self.model1.to(self.device)
        # self.model2.to(self.device)
        # x = x.to(self.device)
        self.model1.eval()
        self.model2.eval()
        self.num_classes = torch.tensor(self.num_classes).to(self.device)
        with torch.no_grad():
            output1 = self.model1(x)

            _, pred = torch.max(output1, 1)

            # print(f"Predictions from first model: {pred}")
            if (
                pred == self.num_classes
            ):  # Assuming last class is the one to trigger second model
                output2 = self.model2(x)
                return output2
            else:
                return output1
