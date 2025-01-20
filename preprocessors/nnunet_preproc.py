from nnunetv2.preprocessing.preprocessing import DefaultPreprocessor
import numpy as np

def ct_windowing(image, window_level, window_width):
    """
    Apply CT windowing to a CT image (2D numpy array).

    Parameters:
    - image: 2D numpy array representing the CT scan image.
    - window_level: Window level value (WL).
    - window_width: Window width value (WW).

    Returns:
    - windowed_image: Image after applying CT windowing.
    """
    
    # Compute the lower and upper bounds of the window
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    
    # Clip the image values to the window range
    windowed_image = np.clip(image, lower_bound, upper_bound)
    
    # Normalize the image to the range 0-255 for visualization
    windowed_image = ((windowed_image - lower_bound) / (upper_bound - lower_bound)) * 255
    windowed_image = np.uint8(windowed_image)  # Convert to uint8 for visualization
    
    return windowed_image

class CustomPreprocessor(DefaultPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, data, **kwargs):
        # Apply HU windowing before other preprocessing steps
        # Bassing numbers off of https://kevalnagda.github.io/ct-windowing
        wl = 60
        ww = 300
        data = ct_windowing(data, wl, ww)
        return super().preprocess(data, **kwargs)
