import cv2
from enum import Enum, unique 

"""
    Enumeration class for color space convertion.
"""
@unique 
class ConvertColor(Enum):
    BGR2RGB = 1
    BGR2GRAY = 2
    BGR2HLS = 3
    BGR2HSV = 4
    HLS2BGR = 5
    RGB2BGR = 6
    RGB2GRAY = 7 
    RGB2HLS = 8
    RGB2HSV = 9
    
    """
        Returns the corresponding name and value.
    """
    def describe(self):
        return self.name, self.value
    
"""
    Converts an image to the desired color space.
"""    
def cvtColor(image, cvt_color_enum):
    if cvt_color_enum is ConvertColor.BGR2RGB:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif cvt_color_enum is ConvertColor.BGR2GRAY:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif cvt_color_enum is ConvertColor.BGR2HLS:
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    elif cvt_color_enum is ConvertColor.BGR2HSV:
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif cvt_color_enum is ConvertColor.RGB2BGR:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   
    elif cvt_color_enum is ConvertColor.RGB2GRAY:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif cvt_color_enum is ConvertColor.RGB2HLS:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif cvt_color_enum is ConvertColor.RGB2HSV:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif cvt_color_enum is ConvertColor.HLS2BGR:
        return cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
    else:
        print("Value is not defined.")
        return image
