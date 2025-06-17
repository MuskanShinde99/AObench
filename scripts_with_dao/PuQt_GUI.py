from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import sys
import time

class AdaptiveOpticsGUI(QMainWindow):
    def __init__(self, npupil):
        super().__init__()
        self.setWindowTitle("Adaptive Optics Simulation")
        self.setGeometry(100, 100, 1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        layout = QVBoxLayout()
        
        # Create labels for images
        self.labels = {}
        for name in ["Deformable Mirror", "Phase Screen", "Processed Pyramid Image", "PSF", "Phase Residuals"]:
            label = QLabel(self)
            label.setFixedSize(256, 256)
            layout.addWidget(label)
            self.labels[name] = label
        
        self.central_widget.setLayout(layout)
    
    def update_image(self, name, data):
        if name in self.labels:
            img = self.numpy_to_qimage(data)
            self.labels[name].setPixmap(QPixmap.fromImage(img))
    
    def numpy_to_qimage(self, array):
        height, width = array.shape
        array = (255 * (array - np.min(array)) / (np.max(array) - np.min(array))).astype(np.uint8)
        return QImage(array.data, width, height, width, QImage.Format_Grayscale8)

def run_simulation():
    app = QApplication(sys.argv)
    npupil = 256
    window = AdaptiveOpticsGUI(npupil)
    window.show()
    
    # Simulate updates
    for _ in range(50):
        window.update_image("Deformable Mirror", np.random.rand(npupil, npupil))
        window.update_image("Phase Screen", np.random.rand(npupil, npupil))
        window.update_image("Processed Pyramid Image", np.random.rand(npupil, npupil))
        window.update_image("PSF", np.random.rand(npupil, npupil))
        window.update_image("Phase Residuals", np.random.rand(npupil, npupil))
        app.processEvents()
        time.sleep(0.1)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_simulation()
