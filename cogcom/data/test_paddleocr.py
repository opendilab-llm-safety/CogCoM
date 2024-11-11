import sys
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

def main(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    result = ocr.ocr(img_array, cls=True)
    print(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_paddleocr.py <image_path>")
    else:
        main(sys.argv[1])