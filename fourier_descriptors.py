# pour chaque masques dans le dossier
# recuperer les contours
# applique transformee de fourier dessus
# nb : plus on "oublie" les hautes frequences plus on perd des details (--> low-pass filter)

import cv2
import numpy as np
from matplotlib import pyplot as plt

MASK_FILE = "data/masks/0.png"

mask = cv2.imread(MASK_FILE)

#cv2.imshow("mask", mask)
#cv2.waitKey(0)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_array = np.squeeze(np.array(contours))
print(f"shape contours :{contour_array.shape}")

complex_contour = contour_array[:, 0] + 1j * contour_array[:, 1]

fourier_coeffs = np.fft.fft(complex_contour)
fourier_coeffs = fourier_coeffs[1:] # translation
fourier_coeffs = (np.abs(fourier_coeffs)) # rotation
fourier_coeffs = fourier_coeffs / fourier_coeffs[0] # echelle

feature_vector = fourier_coeffs[:20]



contours_image = np.zeros_like(mask)
cv2.drawContours(contours_image, contours, -1, 255, 2)
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(mask, cmap='gray'), plt.title('Original Image')
plt.subplot(133), plt.imshow(contours_image, cmap='gray'), plt.title('Contours')



plt.show()

