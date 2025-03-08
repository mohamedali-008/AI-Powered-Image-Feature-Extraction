import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUiType
from harris import myHarris
import matplotlib.image as mpimg
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QPixmap, QPen
from PyQt5.QtCore import Qt


import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QGraphicsPixmapItem

from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QVBoxLayout, QWidget


# Load the UI file and get the main form class
UI_FILE = "main.ui"
Ui_MainWindow, QMainWindowBase = loadUiType(os.path.join(os.path.dirname(__file__), UI_FILE))

class MainApp(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.file_path = ""

        # Connect button signals to slots
        self.openbtn.clicked.connect(self.openFile)
        self.btnGenerate1.clicked.connect(self.generateHarris)
        self.sliderFactor.valueChanged.connect(self.updateFactorLabel)

    def openFile(self):
        # Open a file dialog to select an image file
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)")
        


    def updateFactorLabel(self, value):
        # Update the label text with the current value of the slider
        self.labelFactor.setText(str(value))

        
    def generateHarris(self):
        # Get the current value of the slider and the image file path
        slider_value = self.sliderFactor.value()
        image_path = self.file_path

        # Read the image data
        image_data = mpimg.imread(image_path)

        # Call the function from harris.py
        comp_time = myHarris(image_data, factor=slider_value)
        result = round(comp_time, 2)

        # Update the label displaying computation time
        self.labelTime1.setText(str(result) + "s")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()

    # Show the window in maximized mode
    window.showMaximized()

    sys.exit(app.exec_())
