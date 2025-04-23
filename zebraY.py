import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QFileDialog, QLineEdit, QMessageBox, QProgressBar, QComboBox, QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

polygon_points = []


def rotate_if_needed(frame):
    h, w = frame.shape[:2]
    if h > w:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def roi_click_event(event, x, y, flags, param):
    global polygon_points, roi_img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        cv2.circle(roi_img_copy, (x, y), 3, (0, 0, 255), -1)
        if len(polygon_points) > 1:
            cv2.polylines(roi_img_copy, [np.array(polygon_points)], isClosed=False, color=(255, 0, 0), thickness=2)
        cv2.imshow("Select Area Polygon", roi_img_copy)

def select_roi_polygon(image):
    global polygon_points, roi_img_copy
    polygon_points = []
    image = rotate_if_needed(image)
    roi_img_copy = image.copy()
    cv2.namedWindow("Select Area Polygon")
    cv2.setMouseCallback("Select Area Polygon", roi_click_event)
    print("Area için noktaları tıklayın. Enter ile onaylayın, Esc ile iptal.")
    while True:
        cv2.imshow("Select Area Polygon", roi_img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break
        if key == 27:
            polygon_points = []
            break
    cv2.destroyAllWindows()
    return np.array(polygon_points, dtype=np.int32) if polygon_points else None

def select_reference_points(image):
    points = []
    image = rotate_if_needed(image)
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Reference Points", image)
    clone = image.copy()
    cv2.imshow("Select Reference Points", clone)
    cv2.setMouseCallback("Select Reference Points", click_event)
    print("Referans uzunluk için iki noktaya tıklayın. Enter ile onayla, Esc ile iptal.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break
        elif key == 27:
            points = []
            break
    cv2.destroyAllWindows()
    return points

class WorkerThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str, str)

    def __init__(self, video_path, cm_per_pixel, aquarium_type, roi_polygon=None):
        super().__init__()
        self.video_path = video_path
        self.cm_per_pixel = cm_per_pixel
        self.aquarium_type = aquarium_type
        self.roi_polygon = roi_polygon

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, background_frame = cap.read()
        if not ret:
            self.finished.emit("", "Failed to open video.")
            return

        background_frame = rotate_if_needed(background_frame)
        background_rgb = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
        height, width, _ = background_rgb.shape

        if self.roi_polygon is not None:
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [self.roi_polygon], 255)
        else:
            roi_mask = None

        fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=100, detectShadows=False)
        detections = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        heatmap = np.zeros((height, width), dtype=np.float32)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = rotate_if_needed(frame)
            frame_index += 1
            time_sec = frame_index / fps

            roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            fgmask_img = fgbg.apply(gray)
            fgmask_img[cv2.inRange(gray, 250, 255) > 0] = 0
            
            
            reflection_mask = cv2.inRange(gray, 250, 255)
            fgmask_img[reflection_mask > 0] = 0
            
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask_img = cv2.morphologyEx(fgmask_img, cv2.MORPH_OPEN, kernel)
            fgmask_img = cv2.morphologyEx(fgmask_img, cv2.MORPH_CLOSE, kernel)


            contours, _ = cv2.findContours(fgmask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > 100:
                    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                    if 5 < radius < 80:
                        # Eğer Y akvaryumu ise, doğrulama için poligon testi yapalım
                        if self.roi_polygon is not None and cv2.pointPolygonTest(self.roi_polygon, (cx, cy), False) < 0:
                            continue
                        quadrant = self.get_quadrant(cx, cy, width, height)
                        # Tespitlerde sabit radius değeri (örneğin 10) kullanılıyor.
                        detections.append((time_sec, int(cx), int(cy), 10, quadrant))

            self.progress_updated.emit(int((frame_index / total_frames) * 100))

        cap.release()
        if len(detections) < 2:
            self.finished.emit("", "Not enough detections.")
            return

        travel_distance = 0.0
        travel_time = 0.0
        rest_time = 0.0
        for i in range(1, len(detections)):
            t1, x1, y1, _, _ = detections[i-1]
            t2, x2, y2, _, _ = detections[i]
            dist = math.hypot(x2 - x1, y2 - y1)
            dt = t2 - t1
            if dist > 3:
                travel_distance += dist
                travel_time += dt
            else:
                rest_time += dt
        avg_speed = (travel_distance / travel_time) * self.cm_per_pixel if travel_time > 0 else 0
        travel_distance_cm = travel_distance * self.cm_per_pixel
        avg_speed_cm = avg_speed * self.cm_per_pixel
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]


        for (_, cx, cy, radius, _) in detections:
            mask = np.zeros_like(heatmap, dtype=np.float32)
            cv2.circle(mask, (cx, cy), radius, 0.5, thickness=-1)
            heatmap += mask
        masked_heatmap = np.ma.masked_equal(heatmap, 0)
        cmap = plt.cm.jet.copy()
        cmap.set_bad(color='none')
        
        upper_left_time = upper_right_time= bottom_left_time = bottom_right_time= top_time = bottom_time = 0.0
        valid_detections = sorted(detections, key=lambda x: x[0])
        x, y, w, h = cv2.boundingRect(self.roi_polygon)
        mid_x = x + w / 2
        mid_y = y + h / 2
        if self.aquarium_type.lower() == 'y':
            # Y akvaryum: Üst kısmı ikiye (sol ve sağ), alt kısmı tek bölge (bottom)
            top_boundary = height / 2
            
            for i in range(len(valid_detections)-1):
                t_curr, cx, cy, _, _ = valid_detections[i]
                t_next = valid_detections[i+1][0]
                dt = t_next - t_curr
                if cy < mid_y:
                    if cx < mid_x / 2:
                        upper_left_time += dt
                    else:
                        upper_right_time += dt
                else:
                    bottom_time += dt
            region_info = (
                f"Y-Aquarium Analysis:\n"
                f"Upper Left Duration (s): {upper_left_time:.2f}\n"
                f"Upper Right Duration (s): {upper_right_time:.2f}\n"
                f"Top Total Duration (s): {upper_left_time + upper_right_time:.2f}\n"
                f"Bottom Duration (s): {bottom_time:.2f}"
            )
        else:
            # Normal akvaryum: Ekranı dikeyde sol ve sağ bölgeye ayır.
            left_time = right_time = 0.0
            top_boundary = height / 2
            for i in range(len(valid_detections)-1):
                t_curr, cx, cy, _, _ = valid_detections[i]
                t_next = valid_detections[i+1][0]
                dt = t_next - t_curr
                
                if cy < mid_y:
                    if cx < mid_x:
                        upper_left_time += dt
                    else:
                        upper_right_time += dt
                else:
                    if cx < mid_x:
                        bottom_left_time += dt
                    else:
                        bottom_right_time += dt
            region_info = (
                f"Normal Aquarium Analysis:\n"
                f"Upper Left Duration (s): {upper_left_time:.2f}\n"
                f"Upper Right Duration (s): {upper_right_time:.2f}\n"
                f"Bottom Left Duration (s): {bottom_left_time:.2f}\n"
                f"Bottom Right Duration (s): {bottom_right_time:.2f}\n"
                f"Top Total Duration (s): {upper_left_time + upper_right_time:.2f}\n"
                f"Bottom Total Duration (s): {bottom_left_time + bottom_right_time:.2f}"
                
            )
        
    
        plt.figure()
        plt.imshow(background_rgb)
        plt.imshow(masked_heatmap, cmap=cmap, alpha=0.7)
        plt.title("Fish Movement Heatmap")
        plt.colorbar(label="Intensity")
        output_img = f"{base_name}_heatmap.jpg"
        plt.savefig(output_img)
        plt.close()
        # plt.show()
        
        video_basename = os.path.basename(self.video_path)
        file_stem = os.path.splitext(video_basename)[0]
        output_csv = f"{file_stem}_results.csv"
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([])
            writer.writerow(["Total Distance (cm)", f"{travel_distance_cm:.2f}"])
            writer.writerow(["Average Speed (cm/s)", f"{avg_speed_cm:.2f}"])
            writer.writerow(["Movement Duration (s)", f"{travel_time:.2f}"])
            writer.writerow(["Rest Duration (s)", f"{rest_time:.2f}"])
            writer.writerow([])
            if self.aquarium_type.lower() == 'y':
                writer.writerow(["Upper Left Duration (s)", f"{upper_left_time:.2f}"])
                writer.writerow(["Upper Right Duration (s)", f"{upper_right_time:.2f}"])
                writer.writerow(["Top Total Duration (s)", f"{upper_left_time + upper_right_time:.2f}"])
                writer.writerow(["Bottom Duration (s)", f"{bottom_time:.2f}"])
                
            else:
                writer.writerow(["Upper Left Duration (s)", f"{upper_left_time:.2f}"])
                writer.writerow(["Upper Right Duration (s)", f"{upper_right_time:.2f}"])
                writer.writerow(["Bottom Left Duration (s)", f"{bottom_left_time:.2f}"])
                writer.writerow(["Bottom Right Duration (s)", f"{bottom_right_time:.2f}"])
                writer.writerow(["Top Total Duration (s)", f"{upper_left_time + upper_right_time:.2f}"])
                writer.writerow(["Bottom Total Duration (s)", f"{bottom_left_time + bottom_right_time:.2f}"])
            writer.writerow([])
            

        self.finished.emit(output_csv, f"Analysis complete!\nDistance: {travel_distance_cm:.2f} cm\nSpeed: {avg_speed:.2f} cm/s\nResults saved in {output_csv}\nHeatmap saved in {output_img}")
        
    def get_quadrant(self, cx, cy, width, height):
        if cx < width / 2 and cy < height / 2:
            return "Upper Left"
        elif cx >= width / 2 and cy < height / 2:
            return "Upper Right"
        elif cx < width / 2 and cy >= height / 2:
            return "Lower Left"
        else:
            return "Lower Right"  
        

class VideoAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fish Tracker")
        self.video_path = ""
        self.cm_per_pixel = None
        self.aquarium_type = 'n'
        self.roi_polygon = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.file_label = QLabel("No video selected.")
        self.select_button = QPushButton("Select Video")
        self.preview_label = QLabel()
        self.preview_label.setFixedHeight(150)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.aquarium_type_combo = QComboBox()
        self.aquarium_type_combo.addItems(["Normal", "Y-shaped"])
        self.roi_button = QPushButton("Select Area Polygon")
        self.ref_button = QPushButton("Select Physical Reference")
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.run_button = QPushButton("Start Analysis")

        layout.addWidget(self.file_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.preview_label)
        layout.addWidget(QLabel("Aquarium Type:"))
        layout.addWidget(self.aquarium_type_combo)
        layout.addWidget(self.roi_button)
        layout.addWidget(self.ref_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

        self.select_button.clicked.connect(self.select_file)
        self.roi_button.clicked.connect(self.select_roi)
        self.ref_button.clicked.connect(self.select_reference)
        self.run_button.clicked.connect(self.run_analysis)

    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if fname:
            self.video_path = fname
            self.file_label.setText(fname)
            cap = cv2.VideoCapture(fname)
            ret, frame = cap.read()
            cap.release()
            if ret:
                h, w = frame.shape[:2]
                if h > w:
                    # Dikey video ise saat yönünde çevir (yataylaştır)
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaledToHeight(150, Qt.SmoothTransformation)
                self.preview_label.setPixmap(pixmap)

    def select_roi(self):
        if not self.video_path:
            return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.roi_polygon = select_roi_polygon(frame)

    def select_reference(self):
        if not self.video_path:
            return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            points = select_reference_points(frame)
            if len(points) == 2:
                px_dist = math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
                dist, ok = QInputDialog.getDouble(self, "Real-world Distance", "Enter known distance (cm):", decimals=2)
                if ok and dist > 0:
                    self.cm_per_pixel = dist / px_dist

    def run_analysis(self):
        if not self.video_path or self.cm_per_pixel is None:
            QMessageBox.warning(self, "Error", "Video or reference not set.")
            return
        self.aquarium_type = 'y' if self.aquarium_type_combo.currentText().lower().startswith('y') else 'n'
        self.thread = WorkerThread(self.video_path, self.cm_per_pixel, self.aquarium_type, self.roi_polygon)
        self.thread.progress_updated.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.analysis_finished)
        self.thread.start()

    def analysis_finished(self, _, message):
        QMessageBox.information(self, "Analysis Result", message)
        self.progress_bar.setValue(100)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoAnalyzerApp()
    win.show()
    sys.exit(app.exec_())