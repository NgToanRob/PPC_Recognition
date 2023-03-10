import cv2
import numpy as np
import os
# Figure 5
# (A) raw image
image = cv2.imread('2mm/1B_0001_1.png')
# print(image.shape)
cv2.imshow('Raw Image', image)
# (B) intercept of the caliper;
caliper = image[100:700, 0:30]
caliper = cv2.copyMakeBorder(caliper,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
cv2.imshow('Cropped Image', caliper)

# (C) conversion to gray-scale;
gray_caliper = cv2.cvtColor(caliper, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray scaled Image', gray_caliper)


#  (D) edge detection by the Laplace operator;
laplacian = cv2.Laplacian(gray_caliper, cv2.CV_8UC1, ksize=3)
cv2.imshow('Egde detected Image', laplacian)

# (E) binarization from gray-scale to monochrome;
threshold = 127
binarized = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('Binarize with threshold Image', binarized)
print(254 in binarized)

#  (F) morphological open operation.
# kernel = np.ones((3, 3), np.uint8)
# morphological_open = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
# cv2.imshow('Morphological Opened Image', morphological_open)


# Figure 6

# 6A find and draw contours
contours, hierarchy = cv2.findContours(
    binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"number contours: {len(contours)}")
caliper_copy = caliper.copy()
cv2.drawContours(image=caliper_copy, contours=contours, contourIdx=-1,
                 color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
cv2.imshow('Find Contours', caliper_copy)

# 6B Filter out large contours
threshold = 61
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < threshold]
# filtered_contours = [cnt for cnt in contours if len(cnt) < threshold]
print(f"number contours after filtering out by pixel of contour: {len(filtered_contours)}")
caliper_copy = caliper.copy()
cv2.drawContours(image=caliper_copy, contours=filtered_contours, contourIdx=-1, color=(0,0,255), thickness=1)
cv2.imshow(f'Filter out Contours with threshold {threshold} ', caliper_copy)




# 6C filter out the small and medium contours
caliper_copy = caliper.copy()
contours = filtered_contours
# cv2.drawContours(image=caliper_copy, contours=contours[0], contourIdx=-1, color=(0,0,255), thickness=1)
# cv2.imshow(f'sssssssss{threshold} ', caliper_copy)
# print(cv2.pointPolygonTest(contours[0], (11, 602), False))

# print(len(contours))
# Get the height of the image
height, width = caliper_copy.shape[:2]
print(f'image shape {height,width}')
# Initialize the maximum count of intersecting contours and the y-axis with the maximum count
max_count = 0
filtered_contours = []

# Traverse all y-axes in the image
for x in range(width):
    count = 0
    index_of_filtered_cnt = []
    for y in range(height):

        # Check if each contour intersects with the current y-axis
        for i, cnt in enumerate(contours):
            if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                if i not in index_of_filtered_cnt: index_of_filtered_cnt.append(i)
                count += 1
    # Update the maximum count and y-axis if a greater count is found
    if count > max_count:
        max_count = count
        filtered_contours = [contours[i] for i in index_of_filtered_cnt]

print(f'number contours after filtering out by max point in contour: {len(filtered_contours)}')
cv2.drawContours(image=caliper_copy, contours=filtered_contours, contourIdx=-1, color=(0,0,255), thickness=1)
cv2.imshow(f'Filter out Contours with max intersect', caliper_copy)



# 6D filtering out inner contours
# Sort the contours by the upper left corner position
contours = sorted(filtered_contours, key=lambda x: x[0][0][1])

# Initialize a list to store the pixel distance between adjacent contours
pixel_distance = []
bounding_boxes = []


# Loop through the sorted contours to calculate the pixel distance
box2 = []
for i in range(1, len(contours)):
    box1 = np.int0(cv2.boxPoints(cv2.minAreaRect(contours[i-1])))
    x1, y1 = box1[0]
    box2 = np.int0(cv2.boxPoints(cv2.minAreaRect(contours[i])))
    x2, y2 = box2[0]
    pixel_distance.append(y2 - y1)
    bounding_boxes.append(box1)
bounding_boxes.append(box2)
# Find the mode of the pixel distance
# mode = max(pixel_distance)
print(pixel_distance)



# # visualize
# caliper_copy = caliper.copy()
# for box in bounding_boxes:
#     print(box)
#     caliper_copy = cv2.rectangle(caliper_copy, box[0], box[-2], (0,255,255), 1)
# cv2.imshow('scales of the caliper', caliper_copy)



# # Destroy
cv2.waitKey(0)
cv2.destroyAllWindows()


# Create class to extract calipiuer from source image