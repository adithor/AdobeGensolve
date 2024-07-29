import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint

class ShapeCorrectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Shape Correction')
        self.canvas = QPixmap(800, 600)
        self.canvas.fill(Qt.white)
        self.label = QLabel(self)
        self.label.setPixmap(self.canvas)
        self.setCentralWidget(self.label)
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.label.pixmap())
            painter.setPen(QPen(Qt.black, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.correct_shapes()

    def correct_shapes(self):
        image = self.label.pixmap().toImage()
        buffer = image.bits().asstring(image.byteCount())
        img = np.frombuffer(buffer, dtype=np.uint8).reshape((image.height(), image.width(), 4))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        painter = QPainter(self.label.pixmap())
        painter.setPen(QPen(Qt.red, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            if len(approx) == 2:  # Straight line
                line = approx.reshape(-1, 2)
                painter.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
            elif len(approx) == 3:  # Triangle
                points = [QPoint(point[0][0], point[0][1]) for point in approx]
                painter.drawPolygon(*points)
            elif len(approx) == 4:  # Rectangle or square
                points = [QPoint(point[0][0], point[0][1]) for point in approx]
                painter.drawPolygon(*points)
            else:
                area = cv2.contourArea(contour)
                bounding_rect = cv2.boundingRect(contour)
                aspect_ratio = float(bounding_rect[2]) / bounding_rect[3]

                if 0.9 <= aspect_ratio <= 1.1:  # Circle
                    center, radius = cv2.minEnclosingCircle(contour)
                    painter.drawEllipse(QPoint(int(center[0]), int(center[1])), int(radius), int(radius))
                else:
                    points = [QPoint(point[0][0], point[0][1]) for point in approx]
                    painter.drawPolygon(*points)

        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ShapeCorrectionApp()
    ex.show()
    sys.exit(app.exec_())
