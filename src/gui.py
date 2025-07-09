import sys
import numpy as np
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QRubberBand, QPushButton,QSpinBox,QTextEdit
from PyQt6.QtCharts import QChart, QChartView, QLineSeries,QLogValueAxis, QValueAxis, QBarSeries,QBarSet,QScatterSeries
from PyQt6.QtCore import Qt, QRectF , QTimer, QPointF,QThread, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from pyqtgraph import ImageView
import pyqtgraph as pg
import threading
import dao
import matplotlib.pyplot as plt
import time
import toml
from astropy.io import fits
import subprocess
import os
import signal

import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

# ps -fA | grep python
# /usr/lib/qt5/bin/designer

def matplotlib_cmap_to_lut(name: str, n: int = 256):
    cmap = plt.get_cmap(name)
    colors = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.ubyte)
    return colors

def create_image_view_in_placeholder(parent, name: str, cmap: pg.ColorMap) -> ImageView:
    placeholder = parent.findChild(QWidget, name)
    layout = QVBoxLayout(placeholder)
    layout.setContentsMargins(0, 0, 0, 0)

    view = ImageView()
    layout.addWidget(view)

    # Hide buttons and color bar, set colormap
    if hasattr(view, 'ui'):
        view.ui.roiBtn.hide()
        view.ui.menuBtn.hide()
        view.ui.histogram.gradient.hide()
        view.getHistogramWidget().gradient.setColorMap(cmap)

    return view

class ProcessThread(QThread):
    output_received = pyqtSignal(str)  # Signal for stdout/stderr output
    process_finished = pyqtSignal(int)  # Signal when process ends
    process_started = pyqtSignal(int)  # Signal for process PID

    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                ["python", "-u", self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setpgrp
            )

            self.process_started.emit(self.process.pid)  # Émettre PID

            for line in self.process.stdout:
                self.output_received.emit(line.strip())

            for line in self.process.stderr:
                self.output_received.emit(f"ERROR: {line.strip()}")

            self.process.wait()
            self.process_finished.emit(self.process.returncode)

        except Exception as e:
            self.output_received.emit(f"Thread Error: {str(e)}")

        finally:
            self.process = None  # Éviter les références fantômes

    def stop(self):
        if self.process:
            os.killpg(os.getpgid(self.process.pid), 15)  # Envoie SIGTERM
            self.process.wait(timeout=5)  # Attendre que le processus termine proprement
            if self.process is not None and self.process.poll() is None:  # Si encore actif, tuer brutalement
                os.killpg(os.getpgid(self.process.pid), 9)  # SIGKILL en dernier recour


class ProcessManager(QWidget):
    def __init__(self, script_path, start_button=None, stop_button=None, log_output=None):
        super().__init__()
        self.script_path = script_path
        self.process_thread = None

        self.start_button = start_button
        self.stop_button = stop_button
        self.log_output = log_output

        if self.start_button:
            self.start_button.clicked.connect(self.start_process)

        if self.stop_button:
            self.stop_button.clicked.connect(self.stop_process)
            self.stop_button.setEnabled(False)

    def start_process(self):
        if not self.process_thread:
            self.process_thread = ProcessThread(self.script_path)
            self.process_thread.output_received.connect(self.log_output.append)  # Connect stdout/stderr to UI
            self.process_thread.process_finished.connect(self.process_completed)
            self.process_thread.process_started.connect(
                lambda pid: self.log_output.append(f"Started process with PID: {pid}")
            )

            self.process_thread.start()

            self.start_button.setEnabled(False)
            if self.stop_button:
                self.stop_button.setEnabled(True)

    def process_completed(self, exit_code):
        self.log_output.append(f"Process finished with exit code {exit_code}.")
        if self.process_thread:
            self.process_thread.quit()
            self.process_thread.wait()
            self.process_thread = None
        self.start_button.setEnabled(True)
        if self.stop_button:
            self.stop_button.setEnabled(False)

    def stop_process(self):
        if self.process_thread:
            self.process_thread.stop()
            self.process_thread.wait()
            self.log_output.append("Process terminated.")
            self.process_thread = None
            self.start_button.setEnabled(True)
            if self.stop_button:
                self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        if self.process_thread and self.process_thread.isRunning():
            self.stop_process()
            self.process_thread.wait()
        event.accept()

class CustomChartView(QChartView):
    def __init__(self, parent=None, xlabel="x", ylabel="y", n_lines = 1, plot_type="line"):
        chart = QChart()
        super().__init__(chart, parent)
        self.setRenderHint(self.renderHints().Antialiasing)
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.series_list = []
        self.x = None
        self.y_ranges = []
        self.plot_type = plot_type
        self.y_scale_type = "linear"

        self.x_axis = QValueAxis()
        self.y_axis = QValueAxis()

        self.chart().addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
        self.chart().addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)

        self.set_axis_labels(xlabel, ylabel)

        self.flag_reset_zoom = True

        for _ in range(n_lines):
            series = QLineSeries()
            self.chart().addSeries(series)
            series.attachAxis(self.x_axis)
            series.attachAxis(self.y_axis)
            self.series_list.append(series)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.lastMousePos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.origin = event.position().toPoint()
            self.rubberBand.setGeometry(self.origin.x(), self.origin.y(), 0, 0)
            self.rubberBand.setStyleSheet("background-color: rgba(200, 200, 200, 100);")
            self.rubberBand.show()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.position() - self.lastMousePos
            self.chart().scroll(-delta.x(), delta.y())
            self.lastMousePos = event.position()
            self.flag_reset_zoom = False
        elif event.buttons() == Qt.MouseButton.MiddleButton and self.origin:
            current_pos = event.position().toPoint()
            x = min(self.origin.x(), current_pos.x())
            y = min(self.origin.y(), current_pos.y())
            width = abs(current_pos.x() - self.origin.x())
            height = abs(current_pos.y() - self.origin.y())
            self.rubberBand.setGeometry(x, y, width, height)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.MiddleButton and self.origin:
            rect = self.rubberBand.geometry()
            if rect.width() > 10 and rect.height() > 10:
                self.chart().zoomIn(QRectF(rect))
                self.flag_reset_zoom = False
            self.rubberBand.hide()

    def wheelEvent(self, event: QWheelEvent):
        zoomFactor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.chart().zoom(zoomFactor)
        self.flag_reset_zoom = False

    def set_axis_labels(self, xlabel, ylabel):
        self.x_axis.setTitleText(xlabel)
        self.y_axis.setTitleText(ylabel)

    def draw(self, datasets):
        if not all(isinstance(ds, (list, tuple)) and len(ds) == 2 for ds in datasets):
            raise ValueError("Datasets must be a list of (x, y) tuples or lists.")

        self.x = datasets[0][0] if datasets else None
        self.y_ranges = []

        for i, (x, y) in enumerate(datasets):
            if self.plot_type == "stem":
                lines = []
                for px, py in zip(x.squeeze(), y.squeeze()):
                    lines.append(QPointF(px, 0))  # Start from zero
                    lines.append(QPointF(px, py))  # Go to actual y-value
                    lines.append(QPointF(float('nan'), float('nan')))

                self.series_list[i].replace(lines)  # Update line series

            else:
                points = [QPointF(px, py) for px, py in zip(x.squeeze(), y.squeeze())]
                self.series_list[i].replace(points)

            self.y_ranges.append((min(y), max(y)))

        if self.flag_reset_zoom:
            self.resetZoom()

    def set_log_scale(self, x_log=True, y_log=True):
        x_label = self.x_axis.titleText()
        y_label = self.y_axis.titleText()
        self.y_scale_type = "log"
        self.chart().removeAxis(self.x_axis)
        self.chart().removeAxis(self.y_axis)

        self.x_axis = QLogValueAxis() if x_log else QValueAxis()
        self.y_axis = QLogValueAxis() if y_log else QValueAxis()

        self.x_axis.setTitleText(x_label)
        self.y_axis.setTitleText(y_label)

        self.chart().addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
        self.chart().addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)

        for series in self.series_list:
            series.attachAxis(self.x_axis)
            series.attachAxis(self.y_axis)

    def set_legend(self,legend):
        for i in range(len(self.series_list)):
            self.series_list[i].setName(legend[i])


    def resetZoom(self):
        axes = self.chart().axes()
        if axes:
            x_axis = axes[0]
            y_axis = axes[1]

            if self.x is not None and self.y_ranges:
                xRange = np.min(self.x), (np.max(self.x))
                yMin = min(y_range[0] for y_range in self.y_ranges)
                yMax = max(y_range[1] for y_range in self.y_ranges)
                y_abs_max = max([np.abs(yMin),yMax])
                x_axis.setRange(xRange[0], xRange[1])
                if self.y_scale_type == "linear":
                    y_axis.setRange(-y_abs_max, y_abs_max)
                else :
                    if yMin.ndim > 0:
                        yMin = yMin[0]
                    if yMax.ndim > 0:
                        yMax = yMax[0]
                    y_axis.setRange(yMin, yMax)

        self.flag_reset_zoom = True
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("gui.ui", self)
        self.init_images()
        self.init_vector_plots()
        self.init_shm()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_images)
        self.timer.start(100)
        self.sem_nb = 9
    def init_shm(self):
        with open('shm_path.toml', 'r') as f:
            shm_path = toml.load(f)
        self.slopes_image_shm             = dao.shm(shm_path['slopes_image_shm'])
        self.phase_screen_shm             = dao.shm(shm_path['phase_screen_shm'])
        self.dm_phase_shm                 = dao.shm(shm_path['dm_phase_shm'])
        self.phase_residuals_shm          = dao.shm(shm_path['phase_residuals_shm'])
        self.normalized_psf_shm           = dao.shm(shm_path['normalized_psf_shm'])
        self.commands_shm                 = dao.shm(shm_path['commands_shm'])
        self.residual_modes_shm           = dao.shm(shm_path['residual_modes_shm'])
        self.computed_modes_shm           = dao.shm(shm_path['computed_modes_shm'])
        self.dm_kl_modes_shm              = dao.shm(shm_path['dm_kl_modes_shm'])

    def init_vector_plots(self):
        self.computed_KL_modes_view = CustomChartView(self.computed_KL_modes_widget,"mode","amplitude", n_lines = 2)
        layout = QVBoxLayout(self.computed_KL_modes_widget)
        layout.addWidget(self.computed_KL_modes_view)
        self.computed_KL_modes_view.set_legend(["residual modes","computed modes"])
        self.reset_computed_KL_modes_view_button.clicked.connect(self.computed_KL_modes_view.resetZoom)

        self.computed_act_pos_view = CustomChartView(self.computed_act_pos_widget,"mode","amplitude [2 pi rad ptp]")
        layout = QVBoxLayout(self.computed_act_pos_widget)
        layout.addWidget(self.computed_act_pos_view)
        self.computed_act_pos_view.set_legend(["act pos projeted"])
        self.reset_computed_act_pos_view_button.clicked.connect(self.computed_act_pos_view.resetZoom)

        self.commands_view = CustomChartView(self.commands_widget,"mode","amplitude [2 pi rad ptp]")
        layout = QVBoxLayout(self.commands_widget)
        layout.addWidget(self.commands_view)
        self.commands_view.set_legend(["commands"])
        self.reset_commands_view_button.clicked.connect(self.commands_view.resetZoom)



    def init_images(self):
        # Generate LUT and ColorMap once
        lut = matplotlib_cmap_to_lut('viridis')
        cmap = pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut)

        # All image views and their corresponding placeholder names
        placeholders = {
            "slopes_image_widget": "slopes_image_view",
            "phase_screen_widget": "phase_screen_view",
            "dm_phase_widget": "dm_phase_view",
            "phase_residuals_widget": "phase_residuals_view",
            "normalized_psf_widget": "normalized_psf_view",
            "cam_1_widget": "cam_1_view",
            "cam_2_widget": "cam_2_view",
        }

        for placeholder_name, attr_name in placeholders.items():
            view = create_image_view_in_placeholder(self, placeholder_name, cmap)
            setattr(self, attr_name, view)

    def update_images(self):
        data1 = np.random.rand(100, 100)
        self.pyramid_view.setImage(data1, autoLevels=False,autoRange=False)


        # self.slopes_image_view.setImage(self.slopes_image_shm.get_data(check=False, semNb=self.sem_nb), autoLevels=(self.autoscale_slopes_image_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
        # self.phase_screen_view.setImage(self.phase_screen_shm.get_data(check=False, semNb=self.sem_nb), autoLevels=(self.autoscale_phase_screen_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
        # self.dm_phase_view.setImage(self.dm_phase_shm.get_data(check=False, semNb=self.sem_nb), autoLevels=(self.autoscale_dm_phase_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
        # self.phase_residuals_view.setImage(self.phase_residuals_shm.get_data(check=False,semNb=self.sem_nb),autoLevels=(self.autoscale_phase_residuals_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
        # self.normalized_psf_view.setImage(self.normalized_psf_shm.get_data(check=False,semNb=self.sem_nb), autoLevels=(self.autoscale_normalized_psf_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)



        self.slopes_image_view.setImage(fits.getdata("../outputs/GUI_tests/slopes_image.fits"), autoLevels=(self.autoscale_slopes_image_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
        self.phase_screen_view.setImage(fits.getdata("../outputs/GUI_tests/phase_screen.fits"), autoLevels=(self.autoscale_phase_screen_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
        self.dm_phase_view.setImage(fits.getdata("../outputs/GUI_tests/dm_phase.fits"), autoLevels=(self.autoscale_dm_phase_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
        self.phase_residuals_view.setImage(fits.getdata("../outputs/GUI_tests/phase_residuals.fits"),autoLevels=(self.autoscale_phase_residuals_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)
        self.normalized_psf_view.setImage(fits.getdata("../outputs/GUI_tests/normalized_psf.fits"), autoLevels=(self.autoscale_normalized_psf_checkbox.checkState()==Qt.CheckState.Checked),autoRange=False)

        data2 = np.random.rand(100)-0.5
        if(self.square_computed_KL_modes_checkbox.checkState()==Qt.CheckState.Checked):
            data3 = np.sqrt(np.square(data2))
        else: data3 = data2
        self.computed_KL_modes_view.draw([(np.arange(data3.shape[0]), data3),(np.arange(data3.shape[0]), data3+1)])

        if(self.square_computed_act_pos_checkbox.checkState()==Qt.CheckState.Checked):
            data4 = np.sqrt(np.square(data2))
        else: data4 = data2
        self.computed_act_pos_view.draw([(np.arange(data4.shape[0]), data4)])

        if(self.square_commands_checkbox.checkState()==Qt.CheckState.Checked):
            data5 = np.sqrt(np.square(data2))
        else: data5 = data2
        self.commands_view.draw([(np.arange(data5.shape[0]), data5)])

        # residual_modes = self.residual_modes_shm.get_data(check=False, semNb=self.sem_nb)
        # computed_modes = self.computed_modes_shm.get_data(check=False, semNb=self.sem_nb)
        # if(self.square_computed_KL_modes_checkbox.checkState()==Qt.CheckState.Checked):
        #     residual_modes = np.sqrt(np.square(residual_modes))
        #     computed_modes = np.sqrt(np.square(computed_modes))
        # self.computed_KL_modes_view.draw([(np.arange(residual_modes.shape[0]), residual_modes),(np.arange(computed_modes.shape[0]), computed_modes)])

        
        # dm_kl_modes = self.dm_kl_modes_shm.get_data(check=False, semNb=self.sem_nb)
        # if(self.square_computed_act_pos_checkbox.checkState()==Qt.CheckState.Checked):
        #     dm_kl_modes = np.sqrt(np.square(dm_kl_modes))
        # self.computed_act_pos_view.draw([(np.arange(dm_kl_modes.shape[0]), dm_kl_modes)])

        # commands = self.commands_shm.get_data(check=False, semNb=self.sem_nb)
        # if(self.square_commands_checkbox.checkState()==Qt.CheckState.Checked):
        #     commands = np.sqrt(np.square(commands))
        # self.commands_view.draw([(np.arange(commands.shape[0]), commands)])

    def closeEvent(self, event):
        print("All processes and timers stopped")
        event.accept()

def handle_sigint(signum, frame):
    """Handle Ctrl+C and close the GUI properly"""
    print("SIGINT received, closing application...")
    app.quit()  # Close the application gracefully

if __name__ == "__main__":

    subprocess.run(["python", "setup.py"])

    # Launch the GUI
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, handle_sigint)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

