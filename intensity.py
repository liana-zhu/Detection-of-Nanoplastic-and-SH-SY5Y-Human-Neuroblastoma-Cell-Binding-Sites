import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def load_images_to_array(directory, num):
    images = []
    for i in range(num):
        img_path = os.path.join(directory, "Result of 500ugspike40" + f"{i:03}" + ".tif")
        # img_path = os.path.join(directory, "frame" + str(i + 1) + "to" + str(i + 2) + ".png")
        with Image.open(img_path) as img:
            arr = np.array(img)
            images.append(arr)
    return np.array(images)

def calculateIntensities(startTuple, endTuple, fromFrame=0):
    pixelMean = images[fromFrame].mean()
    intensities = []
    for i in range(fromFrame, len(images)):
        intensity = 0
        for x in range(startTuple[0], endTuple[0]):
            for y in range(startTuple[1], endTuple[1]):
                intensity += images[i][y][x]

        intensities.append((intensity - pixelMean)/((region_bottom_x - region_top_x) * (region_bottom_y - region_top_y)))

    return intensities

images = load_images_to_array("Tammy/600frames/", 598)

# images = load_images_to_array("Tammy/600frames/", 600)
# figure, (ax2, ax3, ax4, ax5) = plt.subplots(1, 4)


# ax1.imshow(images[0], cmap="gray")
# ax1.title.set_text("Mean Image")
# plt.Normalize(0, 255)


# rectangular region defined by top left to bottom right corner

# region_top_x = 314
# region_top_y = 176
# region_bottom_x = 461
# region_bottom_y = 212
# # black center ellipse. FRAME 1.
# # disappears frame 70.
# # white ellipse appears frame 124.
# # original white ellipse fades, but another takes its place
# # new white ellipse stays on surface

# plt.plot(range(len(images)), calculateIntensities((region_top_x, region_top_y), (region_bottom_x, region_bottom_y)))
# plt.title("Black center - Frame 1")
# plt.show()

# region_top_x = 72
# region_top_y = 489
# region_bottom_x = 212
# region_bottom_y = 524
# # black center ellipse. FRAME 1
# # disappears frame 74
# # white ellipse appears frame 102
# # white ellipse stays on surface

# plt.plot(range(len(images)), calculateIntensities((region_top_x, region_top_y), (region_bottom_x, region_bottom_y)))
# plt.title("Black center - Frame 1")
# plt.show()


# region_top_x = 93
# region_top_y = 594
# region_bottom_x = 233
# region_bottom_y = 628
# # white center ellipse. FRAME 118
# # disappears frame 340
# # nothing else appears

# plt.plot(range(118, len(images)), calculateIntensities((region_top_x, region_top_y), (region_bottom_x, region_bottom_y), fromFrame=118))
# plt.title("White center - Frame 118")
# plt.show()

# region_top_x = 458
# region_top_y = 617
# region_bottom_x = 478
# region_bottom_y = 638
# # no ellipse. FRAME 1
# # nothing ever appears

# plt.plot(range(len(images)), calculateIntensities((region_top_x, region_top_y), (region_bottom_x, region_bottom_y)))
# plt.title("No ellipse  - Frame 1")

# # figure.tight_layout()
# plt.show()

(x, y), (minor, major), angle = (274.05389404296875, 14.966105461120605), (19.92026138305664, 89.50321197509766), 89.44464874267578
# sample ellipse detected in frame99to100.png

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
image = np.array(Image.open("frame/frame99to100.png"))
ax1.imshow(image, cmap='gray')

ellipse_points = cv2.ellipse2Poly((int(x), int(y)), (int(minor / 2), int(major / 2)), int(angle), 0, 360, 1)
point_x = []
point_y = []
for point in ellipse_points:
    x = point[0]
    y = point[1]
    if x < 0 or y < 0 or x > image.shape[1] or y > image.shape[0]:
        continue
    point_x.append(x)
    point_y.append(y)

ax1.plot(point_x, point_y, color='red')

region_top_x = int(x - major/2)
region_top_y = int(y - minor/2)
region_bottom_x = int(x + major/2)
region_bottom_y = int(y + minor/2)

ax2.plot(range(99, len(images)), calculateIntensities((region_top_x, region_top_y), (region_bottom_x, region_bottom_y), 99))
fig.tight_layout()
plt.show()
