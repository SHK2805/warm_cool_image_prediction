import os
from typing import Tuple

import cv2
import numpy as np
from logger.logger import logger

class WarmCoolPredictor:
    def __init__(self, images_dir: str = "images"):
        """
        Initializes the WarmCoolPredictor with a specified directory for images.

        Parameters:
            images_dir (str): The directory path where images are stored.
        """
        self.image_dir_path = images_dir

    def delete_files(self):
        """
        Deletes all files in the images directory.
        """
        for file in os.listdir(self.image_dir_path):
            os.remove(os.path.join(self.image_dir_path, file))
        logger.info("All files in the image directory have been deleted.")

    def get_files(self):
        """
        Retrieves file names in the images directory with specified extensions.

        Returns:
            List[str]: A list of file names with specified extensions.
        """
        allowed_extensions = ('.jpg', '.jpeg', '.png', '.gif')
        return [file for file in os.listdir(self.image_dir_path) if file.endswith(allowed_extensions)]

    def get_image(self, image_name: str):
        """
        Reads an image from the images' directory.

        Parameters:
            image_name (str): The name of the image file.

        Returns:
            np.ndarray: The image array.
        """
        return cv2.imread(os.path.join(self.image_dir_path, image_name))

    def get_hsv_image(self, image_name: str):
        """
        Converts an image to HSV color space.

        Parameters:
            image_name (str): The name of the image file.

        Returns:
            np.ndarray: The HSV image array.
        """
        image = self.get_image(image_name)
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def get_hue_channel(self, image_name: str):
        """
        Retrieves the hue channel from an HSV image.

        Parameters:
            image_name (str): The name of the image file.

        Returns:
            np.ndarray: The hue channel array.
        """
        hsv_image = self.get_hsv_image(image_name)
        return hsv_image[:, :, 0]

    def get_saturation_channel(self, image_name: str):
        """
        Retrieves the saturation channel from an HSV image.

        Parameters:
            image_name (str): The name of the image file.

        Returns:
            np.ndarray: The saturation channel array.
        """
        hsv_image = self.get_hsv_image(image_name)
        return hsv_image[:, :, 1]

    def get_value_channel(self, image_name: str):
        """
        Retrieves the value channel from an HSV image.

        Parameters:
            image_name (str): The name of the image file.

        Returns:
            np.ndarray: The value channel array.
        """
        hsv_image = self.get_hsv_image(image_name)
        return hsv_image[:, :, 2]

    def get_dull_percentage(self, image_name: str) -> float:
        """
        Calculates the percentage of dull pixels in an image.

        Parameters:
            image_name (str): The name of the image file.

        Returns:
            float: The percentage of dull pixels.
        """
        saturation_channel = self.get_saturation_channel(image_name)
        value_channel = self.get_value_channel(image_name)
        saturation_threshold = 80
        value_threshold = 100
        dull_mask = (saturation_channel < saturation_threshold) & (value_channel < value_threshold)
        dull_percentage = (np.sum(dull_mask) / saturation_channel.size) * 100
        return dull_percentage

    def get_warm_cool_mask(self, image_name: str):
        """
        Creates masks for warm and cool colors in an image.

        Parameters:
            image_name (str): The name of the image file.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The hue channel, warm mask, and cool mask arrays.
        """
        hue_channel = self.get_hue_channel(image_name)
        warm_mask = hue_channel < 60
        cool_mask = (hue_channel > 90) & (hue_channel <= 180)
        return hue_channel, warm_mask, cool_mask

    def get_warm_cool_percentage(self, image_name: str) -> Tuple[float, float]:
        """
        Calculates the percentages of warm and cool pixels in an image.

        Parameters:
            image_name (str): The name of the image file.

        Returns:
            Tuple[float, float]: The percentages of warm and cool pixels.
        """
        hue_channel, warm_mask, cool_mask = self.get_warm_cool_mask(image_name)
        total_pixels = hue_channel.size
        warm_pixels = np.count_nonzero(warm_mask)
        cool_pixels = np.count_nonzero(cool_mask)
        warm_percentage = (warm_pixels / total_pixels) * 100
        cool_percentage = (cool_pixels / total_pixels) * 100
        return warm_percentage, cool_percentage

    def display_masks(self, image_name: str):
        """
        Displays the hue, warm, and cool masks of an image.

        Parameters:
            image_name (str): The name of the image file.
        """
        hue_channel, warm_mask, cool_mask = self.get_warm_cool_mask(image_name)
        images = cv2.hconcat([hue_channel, warm_mask, cool_mask])
        cv2.imshow("Hue, Warm and Cool Masks", images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict_image(self, image_name: str) -> str:
        """
        Predicts whether the image is warm, cool, dull, or bright.

        Parameters:
            image_name (str): The name of the image file.

        Returns:
            str: The prediction result.
        """
        warm_percentage, cool_percentage = self.get_warm_cool_percentage(image_name)
        dull_percentage = self.get_dull_percentage(image_name)
        logger.info(f"Dull percentage: {dull_percentage:.2f}%")
        logger.info(f"Warm percentage: {warm_percentage:.2f}%")
        logger.info(f"Cool percentage: {cool_percentage:.2f}%")
        self.delete_files()
        is_dull = dull_percentage > 50
        is_warm = warm_percentage > cool_percentage

        if is_warm and is_dull:
            return "Warm and Dull"
        elif is_warm and not is_dull:
            return "Warm and Bright"
        elif not is_warm and is_dull:
            return "Cool and Dull"
        else:
            return "Cool and Bright"

    def loop_images(self):
        """
        Loops through the images in the images directory and predicts each image.
        """
        for image_name in self.get_files():
            logger.info(f"Predicting image: {image_name}")
            prediction = self.predict_image(image_name)
            logger.info(f"The image {image_name} is: {prediction}")
            logger.info("*" * 50)

if __name__ == "__main__":
    wc_predictor = WarmCoolPredictor("../images")
    wc_predictor.loop_images()
