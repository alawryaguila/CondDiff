import torch
import torch.nn.functional as F


class PSNRMetric:
    def __init__(self, max_val=1.0):
        """
        Initialize the PSNR metric.

        Args:
            max_val (float): The maximum possible pixel value of the image. Default is 1.0 for normalized images.
        """
        self.max_val = max_val

    def calculate_mse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate Mean Squared Error (MSE) between the true and predicted images.

        Args:
            y_true (torch.Tensor): Ground truth images.
            y_pred (torch.Tensor): Predicted images.

        Returns:
            torch.Tensor: MSE value.
        """
        return F.mse_loss(y_true, y_pred)

    def calculate_psnr(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between the true and predicted images.

        Args:
            y_true (torch.Tensor): Ground truth images.
            y_pred (torch.Tensor): Predicted images.

        Returns:
            torch.Tensor: PSNR value in dB.
        """
        mse = self.calculate_mse(y_true, y_pred)

        if mse == 0:
            return float('inf')  # PSNR is infinite if MSE is 0

        psnr_value = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr_value

    def calculate_psnr_mask(self, y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between the true and predicted images, 
        considering only the regions where the mask is zero.

        Args:
            y_true (torch.Tensor): Ground truth images.
            y_pred (torch.Tensor): Predicted images.
            mask (torch.Tensor): Mask indicating the regions to be considered (0 for masked regions, 1 for valid regions).

        Returns:
            torch.Tensor: PSNR value in dB.
        """
        # Only consider the regions where mask is 0 (valid regions)
        valid_pixels = (1 - mask).to(torch.bool)  # valid pixels are where mask is 0

        # Calculate the squared error only for the valid pixels
        mse = torch.sum((y_true - y_pred) ** 2 * valid_pixels) / valid_pixels.sum()

        if mse == 0:
            return float('inf')  # PSNR is infinite if MSE is 0

        psnr_value = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr_value