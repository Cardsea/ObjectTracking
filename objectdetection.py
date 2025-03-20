import cv2
import pyautogui
from PyQt5 import QtWidgets, QtGui
from ultralytics import YOLO
from PIL.ImageQt import ImageQt

pyautogui.FAILSAFE = False

# Load YOLO model
model = YOLO("yolov8n.pt")

# Detect available cameras and get their names
def list_cameras():
    cameras = []
    for i in range(3):  # Limit to available devices (0-2)  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append((i, f"Camera {i}"))  
        cap.release()
    return cameras

available_cameras = list_cameras()
selected_camera = available_cameras[0][0] if available_cameras else -1

# Flags
move_mouse = True
sensitivity = 0.1
list_objects = False

def toggle_move_mouse():
    global move_mouse
    move_mouse = not move_mouse

def toggle_list_objects():
    global list_objects
    list_objects = not list_objects

def apply_settings():
    global selected_camera, sensitivity
    selected_camera = int(window.camera_var.currentText().split()[-1])  
    sensitivity = float(window.sensitivity_var.text())

def select_camera():
    global selected_camera
    selected_camera = int(window.camera_var.currentText().split()[-1])  

# PyQt5 Window
class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Control Panel")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QtWidgets.QVBoxLayout()

        self.camera_var = QtWidgets.QComboBox()
        self.camera_var.addItems([name for _, name in available_cameras])
        self.sensitivity_var = QtWidgets.QLineEdit("0.1")

        self.move_mouse_button = QtWidgets.QPushButton("Move Mouse (Disable with 'k')")
        self.move_mouse_button.clicked.connect(toggle_move_mouse)
        self.layout.addWidget(self.move_mouse_button)

        self.camera_dropdown_label = QtWidgets.QLabel("Select Camera")
        self.layout.addWidget(self.camera_dropdown_label)
        self.layout.addWidget(self.camera_var)

        self.sensitivity_label = QtWidgets.QLabel("Sensitivity")
        self.layout.addWidget(self.sensitivity_label)
        self.layout.addWidget(self.sensitivity_var)

        self.list_objects_button = QtWidgets.QPushButton("List Objects (Toggle)")
        self.list_objects_button.clicked.connect(toggle_list_objects)
        self.layout.addWidget(self.list_objects_button)

        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(apply_settings)
        self.layout.addWidget(self.apply_button)

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        self.layout.addWidget(self.close_button)

        self.cv_frame = QtWidgets.QLabel()
        self.cv_frame.setFixedSize(640, 480)  
        self.layout.addWidget(self.cv_frame)

        self.setLayout(self.layout)

    def update_frame(self, frame):
        if frame is None or frame.isNull():
            print("Frame is null, skipping update!")
            return
        self.cv_frame.setPixmap(QtGui.QPixmap.fromImage(frame))

# Start PyQt5
def start_pyqt():
    global window, app
    app = QtWidgets.QApplication([])
    window = ControlWindow()
    window.show()
    start_opencv()  
    app.exec_()

# OpenCV Object Detection
def start_opencv():
    global selected_camera, move_mouse, sensitivity, list_objects

    if selected_camera == -1:
        print("No camera available.")
        return

    cap = cv2.VideoCapture(selected_camera)

    if not cap.isOpened():
        print(f"Camera {selected_camera} could not be opened.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        best_object = None
        best_conf = 0
        detected_objects = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0]) * 100
                label = result.names[int(box.cls[0])]

                if confidence > best_conf:
                    best_conf = confidence
                    best_object = (x1, y1, x2, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {confidence:.1f}%"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                detected_objects.append(text)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_qt = ImageQt(frame_pil)

        window.update_frame(frame_qt)

        if move_mouse and best_object:
            x1, y1, x2, y2 = best_object
            obj_x = (x1 + x2) // 2
            obj_y = (y1 + y2) // 2

            win_x, win_y, win_w, win_h = window.geometry().getRect()
            mouse_x = win_x + obj_x
            mouse_y = win_y + obj_y

            pyautogui.moveTo(mouse_x, mouse_y, duration=sensitivity)

        if list_objects:
            print("Detected Objects:")
            for obj in detected_objects:
                print(obj)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("k"):
            move_mouse = not move_mouse

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_pyqt()