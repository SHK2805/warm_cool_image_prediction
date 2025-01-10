import cv2
import os

import numpy as np

from logger.logger import logger


class WarmCoolPredictor:
    def __init__(self, images_dir: str = "images"):
        self.image_dir_path = images_dir

    # delete files in the images directory
    def delete_files(self):
        import os
        for file in os.listdir(self.image_dir_path):
            os.remove(os.path.join(self.image_dir_path, file))

    # get the file names in the images directory
    def get_files(self):
        # get files with .jpg, .jpeg, .png, .gif extensions
        return [file for file in os.listdir(self.image_dir_path)
                if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # get the image
    def get_image(self, image_name: str):
        return cv2.imread(os.path.join(self.image_dir_path, image_name))

    # get the hsv image
    def get_hsv_image(self, image_name: str):
        image = self.get_image(image_name)
        # convert the image to hsv
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the hue channel for an image
    def get_hue_channel(self, image_name: str):
        hsv_image = self.get_hsv_image(image_name)
        # this returns a numpy 2D array of the hue channel
        return hsv_image[:, :, 0]

    # get warm and cool mask
    def get_warm_cool_mask(self, image_name: str):
        hue_channel = self.get_hue_channel(image_name)
        # create a mask for warm colors
        warm_mask = hue_channel < 60
        # warm_mask = cv2.inRange(hue_channel, 0, 60)
        # create a mask for cool colors
        cool_mask = (hue_channel > 90) & (hue_channel <= 180)
        # cool_mask = cv2.inRange(hue_channel, 90, 180)
        return hue_channel, warm_mask, cool_mask

    # get warm cool percentage
    def get_warm_cool_percentage(self, image_name: str):
        hue_channel, warm_mask, cool_mask = self.get_warm_cool_mask(image_name)
        # get the total number of pixels in the image
        total_pixels = hue_channel.size
        # get the number of warm pixels
        warm_pixels = np.count_nonzero(warm_mask)
        # get the number of cool pixels
        cool_pixels = np.count_nonzero(cool_mask)
        # calculate the warm percentage
        warm_percentage = (warm_pixels / total_pixels) * 100
        # calculate the cool percentage
        cool_percentage = (cool_pixels / total_pixels) * 100
        return warm_percentage, cool_percentage

    # display the warm, neutral, and cool masks
    def display_masks(self, image_name: str):
        hue_channel, warm_mask, cool_mask = self.get_warm_cool_mask(image_name)
        # display the masks in the same window
        images = cv2.hconcat([hue_channel, warm_mask, cool_mask])
        cv2.imshow("Hue, Warm and Cool Masks", images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # predict the image
    def predict_image(self, image_name: str):
        warm_percentage, cool_percentage = self.get_warm_cool_percentage(image_name)
        logger.info(f"Warm percentage: {warm_percentage: .2f}%")
        logger.info(f"Cool percentage: {cool_percentage: .2f}%")
        if warm_percentage > cool_percentage:
            return "Warm"
        elif cool_percentage > warm_percentage:
            return "Cool"
        else:
            return "Neutral"

    # loop through the images in the images directory
    def loop_images(self):
        for image_name in self.get_files():
            logger.info(f"Predicting image: {image_name}")
            prediction = self.predict_image(image_name)
            logger.info(f"The image {image_name} is: {prediction}")
            logger.info("*" * 50)

if __name__ == "__main__":
    wc_predictor = WarmCoolPredictor()
    wc_predictor.loop_images()

