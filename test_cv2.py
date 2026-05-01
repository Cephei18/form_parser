try:
    import cv2
    print('cv2 version', cv2.__version__)
except Exception as e:
    print('cv2 import failed:', e)
