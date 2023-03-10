import cv2
import numpy as np
import pandas as pd


class CaliperExtractor:
    def __init__(self, image, bounding):
        """
        This function takes in an image and a bounding box and returns an object
        with the image and bounding box as attributes

        @param image The image that the bounding box is drawn on.
        @param bounding An array of 4 numbers that represent the bounding box of the
        object with format `[x_center, y_center, width, height]`
        """
        self.image = image

        self.x, self.y, self.w, self.h = bounding

    def get_PPC(self):
        """
        It takes the image, crops it to the region of interest, converts it to
        grayscale, applies a Laplacian filter, binarizes it, finds the contours,
        filters out the small and medium contours, and then filters out the inner
        contours
        
        @return The PPC and the scaler boxes in input bounding box.
        """
        # code to crop caliper from source image
        caliper = self.image.copy()
        caliper = caliper[(self.y-self.h//2):(self.y+self.h//2),
                  (self.x-self.w//2):(self.x+self.w//2)]

        # Adding black padding by 10 px
        caliper = cv2.copyMakeBorder(
            caliper, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Convert to gray scale
        caliper = cv2.cvtColor(caliper, cv2.COLOR_BGR2GRAY)

        # Edge detection by the Laplace operator;
        caliper = cv2.Laplacian(caliper, cv2.CV_8UC1, ksize=3)

        # Binarization from gray-scale to monochrome
        threshold = 127
        caliper = cv2.threshold(caliper, threshold, 255, cv2.THRESH_BINARY)[1]
        

        # Find and draw contours
        contours, _ = cv2.findContours(
            caliper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Remove large contours
        threshold = 61
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) < threshold]

        # Filter out the small and medium contours
        contours = self._filter_small_and_medium(contours)

        # Filtering out inner contours
        _, ppc = self._get_inner_scaler(contours)

        return ppc

    def _filter_small_and_medium(self, contours):
        """
        It finds the y-axis with the most contours intersecting it, and returns the
        contours that intersect with that y-axis

        This code is used to filter contours in an image. It traverses all y-axes 
        in the image and checks if each contour intersects with the current y-axis. 
        If it does, it adds the contour to a list of filtered contours. The code
        then updates the maximum count and y-axis if a greater count is found 
        and sorts the contours by their upper left corner position. 
        Finally, it returns the filtered contours.

        @param contours The contours that we want to filter.
        @return The contours of the image.
        """
        # Initialize the maximum count of intersecting contours and the y-axis with the maximum count
        max_count = 0
        filtered_contours = []

        # Traverse all y-axes in the image
        for x in range(self.w):
            count = 0
            index_of_filtered_cnt = []
            for y in range(self.h):

                # Check if each contour intersects with the current y-axis
                for i, cnt in enumerate(contours):
                    if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                        if i not in index_of_filtered_cnt:
                            index_of_filtered_cnt.append(i)
                        count += 1
            # Update the maximum count and y-axis if a greater count is found
            if count > max_count:
                max_count = count
                filtered_contours = [contours[i]
                                     for i in index_of_filtered_cnt]
        # Sort the contours by the upper left corner position
        filtered_contours = sorted(filtered_contours, key=lambda x: x[0][0][1])
        return filtered_contours

    def _get_inner_scaler(self, contours):
        """
        The function takes in a list of contours, and returns a list of bounding boxes
        and the pixel to centimeter conversion factor

        This code is calculating the pixel distance between two contours. It first 
        creates two empty lists, pixel_distance and bounding_boxes. It then loops 
        through the sorted contours to calculate the pixel distance between each one. 
        The code then calculates the mean of the pixel distances and multiplies it
        by 5 to get a value for pixels per centimeter (ppc). Finally, it removes 
        any outliers from the list of pixel distances.
        
        @param contours the contours of the image
        @return The bounding boxes and the pixel count per centimeter.
        """
        # Initialize a list to store the pixel distance between adjacent contours
        pixel_distance = []
        bounding_boxes = []

        # Loop through the sorted contours to calculate the pixel distance
        box2 = 0
        for i in range(1, len(contours)):
            box1 = np.int0(cv2.boxPoints(cv2.minAreaRect(contours[i-1])))
            x1, y1 = box1[0]
            box2 = np.int0(cv2.boxPoints(cv2.minAreaRect(contours[i])))
            x2, y2 = box2[0]
            pixel_distance.append(y2 - y1)
            bounding_boxes.append(box1)
        bounding_boxes.append(box2)

        pixel_distance = self._remove_outline(pixel_distance)
        pcc = np.mean(pixel_distance) * 5 # ppc/20mm * 5
        return bounding_boxes, pcc

    def _remove_outline(self, pixel_distance):
        '''
        TODO: find ways to remove the error distances
        '''
        pixel_distance = sorted(pixel_distance)
        return pixel_distance


def rmse(src, pre):
    return np.sqrt(np.mean((src-pre)**2))

if __name__ == '__main__':
    box = [15, 400, 30, 600]

    label = pd.read_excel('label/Ket qua kssg_final_84.xlsx')


    pre_phyX = []
    src_phyX = []

    error_file = []

    for i in range(len(label)):
        filename = label['FileName'][i]
        phyX_label = label['PhyX'][i]

        print(f'file: {filename}')

        src = cv2.imread(f'2mm/{filename}')
        extractor = CaliperExtractor(src, box)
        ppc, scaler_boxes = extractor.get_PPC()
        phyX = 1/ ppc

    # if not np.isnan(phyX):
        src_phyX.append(phyX_label)
        pre_phyX.append(phyX)
    # else:
        # error_file.append(filename)
        # print('Error prediected with file: ', filename)

    pre_phyX = np.array(pre_phyX)
    src_phyX = np.array(src_phyX)

    # print(pre_phyX)

    label['pre_phyX'] = pre_phyX
    print(label['pre_phyX'])
    label.to_excel('predicted.xlsx')
    # print('RMSE: ', rmse(src_phyX, pre_phyX))
    # print('Error predicted: ',  error_file)

