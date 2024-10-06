import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QPushButton, QGraphicsView, QGraphicsScene, QLabel
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

class ImageViewer(QDialog):
    def __init__(self) -> None:
        """Initialize the Image Viewer"""
        super(ImageViewer, self).__init__()
        uic.loadUi('D:/Computer vision/3/P2/GUI MATCH.ui', self)  # Load the UI file

        # Find the buttons, graphics views, and labels from the UI
        self.load_button_1 = self.findChild(QPushButton, 'pushButton')  # Button for loading first image
        self.load_button_2 = self.findChild(QPushButton, 'pushButton_2')  # Button for loading second image
        self.match_button = self.findChild(QPushButton, 'pushButton_3')  # Button for matching images
        self.image_viewer_1 = self.findChild(QGraphicsView, 'graphicsView')  # QGraphicsView for first image
        self.image_viewer_2 = self.findChild(QGraphicsView, 'graphicsView_2')  # QGraphicsView for second image
        self.match_viewer_1 = self.findChild(QGraphicsView, 'graphicsView_3')  # QGraphicsView for matching result
        self.match_viewer_2 = self.findChild(QGraphicsView, 'graphicsView_4')  # First matched template viewer
        self.match_viewer_3 = self.findChild(QGraphicsView, 'graphicsView_5')  # Second matched template viewer
        self.result_label = self.findChild(QLabel, 'label')  # Label to show matching result

        # Create QGraphicsScenes
        self.scene_1 = QGraphicsScene(self)
        self.scene_2 = QGraphicsScene(self)
        self.match_scene_1 = QGraphicsScene(self)
        self.match_scene_2 = QGraphicsScene(self)
        self.match_scene_3 = QGraphicsScene(self)

        # Set the scenes to the graphics views
        self.image_viewer_1.setScene(self.scene_1)
        self.image_viewer_2.setScene(self.scene_2)
        self.match_viewer_1.setScene(self.match_scene_1)
        self.match_viewer_2.setScene(self.match_scene_2)
        self.match_viewer_3.setScene(self.match_scene_3)

        # Connect the buttons to their respective functions
        self.load_button_1.clicked.connect(self.load_first_image)
        self.load_button_2.clicked.connect(self.load_second_image)
        self.match_button.clicked.connect(self.match_images)

        self.img1 = None  # To store the first loaded image
        self.img2 = None  # To store the second loaded image

    def load_first_image(self):
        """Load the first image from file dialog"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open First Image File", "",
                                                    "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        if file_name:
            self.img1 = cv2.imread(file_name)
            if self.img1 is not None:
                self.displayImage(self.img1, self.scene_1)

    def load_second_image(self):
        """Load the second image from file dialog"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Second Image File", "",
                                                    "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        if file_name:
            self.img2 = cv2.imread(file_name)
            if self.img2 is not None:
                self.displayImage(self.img2, self.scene_2)

    def displayImage(self, img, scene):
        """Display the image in the specified QGraphicsScene"""
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qImg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.clear()  # Clear the scene before adding a new image
        scene.addPixmap(pixmap)  # Add the QPixmap to the scene

    def match_images(self):
        """Match the two images and display the results"""
        if self.img1 is not None and self.img2 is not None:
            # Convert images to grayscale
            gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

            # Initialize SIFT detector
            sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

            # Find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)

            # Use Brute Force Matcher
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)

            # Sort matches based on distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Display matching result in the label
            if len(matches) > 10:
                self.result_label.setText("Matching")
            else:
                self.result_label.setText("Not Matching")

            # Draw matches
            matched_img = cv2.drawMatches(self.img1, kp1, self.img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.displayImage(matched_img, self.match_scene_1)  # Show matches in the first match viewer

            # Optionally, show individual keypoints in additional graphics views
            self.show_keypoints(kp1, self.img1, self.match_scene_2)
            self.show_keypoints(kp2, self.img2, self.match_scene_3)

    def show_keypoints(self, keypoints, img, scene):
        """Draw keypoints and display in the specified scene"""
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.displayImage(img_with_keypoints, scene)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())