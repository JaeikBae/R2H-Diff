# import numpy as np
# import cv2

# class HsiTransform(object):
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height

#     def __call__(self, img, type):
#         if type == 'seg': # (h, w)
#             img = img

#         elif type == 'hsi':
#             # 90 degree rotation to the left
#             # (h, w)
            

#         elif type == 'rgb':
#             img = img.transpose((1, 2, 0))

#         img = cv2.resize(img, (self.width, self.height))

#         # Ensure the image has 3 dimensions (add channel if grayscale)
#         if len(img.shape) == 2:  # Grayscale image
#             img = np.expand_dims(img, axis=-1)  # Add channel dimension
        
#         # if type == 'hsi':
#         #     # (h, w, c) -> (c, h, w), use channel as width
#         #     img = np.transpose(img, (2, 0, 1))

#         # # if hsi, (c, h, w) -> (w, c, h), use channel as height
#         # else, (h, w, c) -> (c, h, w)
#         img = np.transpose(img, (2, 0, 1))

#         case = { # normalize data to range [0, 1] -> change to 0~100
#             # 8 classes + background
#             'seg': img / 9.0 * 1.0,
#             'hsi': img / 200.0 * 1.0,
#             'rgb': img / 255.0 * 1.0
#         }
#         img = case[type] # Normalize data to range [0, 1]
#         return img