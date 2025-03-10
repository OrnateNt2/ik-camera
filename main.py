import sys
import platform
import numpy as np
import mvsdk

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QRadioButton, QCheckBox, QSlider,
    QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox,
    QGridLayout, QButtonGroup, QComboBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

def process_mode_30(pair, k):
    """Режим 30 Гц: берём два кадра A, B и считаем B - k*A"""
    if len(pair) < 2:
        return None
    A = pair[-2].astype(np.float32)
    B = pair[-1].astype(np.float32)
    diff = B - k * A
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff

def process_mode_15(group, k):
    """
    Режим 15 Гц: 4 кадра (A, A, B, B)
    Усредняем A_avg, B_avg => diff = B_avg - k * A_avg
    """
    if len(group) < 4:
        return None
    A_avg = (group[-4].astype(np.float32) + group[-3].astype(np.float32)) / 2
    B_avg = (group[-2].astype(np.float32) + group[-1].astype(np.float32)) / 2
    diff = B_avg - k * A_avg
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff

def process_mode_20(group, k):
    """
    Режим 20 Гц: 3 кадра (A, B, C)
    A' = (A + C)/2, diff = B - k*A'
    """
    if len(group) < 3:
        return None
    A_prime = (group[-3].astype(np.float32) + group[-1].astype(np.float32)) / 2
    B = group[-2].astype(np.float32)
    diff = B - k * A_prime
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff

def temporal_median_filter(diff_buffer):
    stacked = np.stack(diff_buffer, axis=0)
    median_filtered = np.median(stacked, axis=0).astype(np.uint8)
    return median_filtered

def apply_threshold(image, threshold_value):
    out = image.copy()
    out[out < threshold_value] = 0
    return out

class AdaptiveK:
    def __init__(self, initial_k=0.95, learning_rate=0.01, target_dark_level=5.0):
        self.k = initial_k
        self.learning_rate = learning_rate
        self.target_dark_level = target_dark_level

    def update(self, dark_frame):
        mean_dark = np.mean(dark_frame)
        error = mean_dark - self.target_dark_level
        self.k -= self.learning_rate * error
        self.k = max(0.5, min(1.5, self.k))
        return self.k


def convert_frame_to_qpixmap(frame):
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        # GRAY
        h, w = frame.shape[:2]
        bytes_per_line = w
        qimg = QImage(
            frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8
        )
    else:
        # BGR -> RGB
        h, w, ch = frame.shape
        rgb = frame[..., ::-1]
        bytes_per_line = ch * w
        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
    return QPixmap.fromImage(qimg)

############################################
#   Главное окно PyQt5
############################################
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demo IR Camera + Выбор разрешения + Режимы + Фильтры")
        
        # -----------------------------
        # Переменные SDK
        # -----------------------------
        self.hCamera = None
        self.monoCamera = False
        self.pFrameBuffer = None
        self.FrameBufferSize = 0
        self.capability = None  # tSdkCameraCapbility
        # Список предустановленных разрешений
        self.resList = []

        # -----------------------------
        # Буферы кадров
        # -----------------------------
        self.frame_buffer = []  # для текущего режима
        self.diff_buffer = []   # для временной медианной фильтрации

        # -----------------------------
        # Параметры режима
        # -----------------------------
        self.mode = 1  # 1=30 Гц, 2=15 Гц, 3=20 Гц

        # -----------------------------
        # Параметры фильтров
        # -----------------------------
        self.use_temporal_filter = False
        self.temporal_window = 3

        self.use_threshold = False
        self.threshold_value = 20

        self.use_adaptive_k = False
        self.base_k = 0.95
        self.adaptive_k = AdaptiveK(initial_k=self.base_k, learning_rate=0.01)

        # Инициализация UI
        self.init_ui()

        # Инициализация камеры
        try:
            self.init_camera()
        except mvsdk.CameraException as e:
            print("CameraInit failed:", e)
            sys.exit(1)

        # Запускаем таймер на ~30 мс
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):

        # --- Группа "Режим (частота фонарика)" ---
        self.radio_30 = QRadioButton("30 Гц (2 кадра)")
        self.radio_15 = QRadioButton("15 Гц (4 кадра)")
        self.radio_20 = QRadioButton("20 Гц (3 кадра)")
        self.radio_30.setChecked(True)

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_30, 1)
        self.mode_group.addButton(self.radio_15, 2)
        self.mode_group.addButton(self.radio_20, 3)
        self.mode_group.buttonClicked[int].connect(self.on_mode_changed)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.radio_30)
        mode_layout.addWidget(self.radio_15)
        mode_layout.addWidget(self.radio_20)

        mode_group_box = QGroupBox("Режим (частота)")
        mode_group_box.setLayout(mode_layout)

        # --- Группа "Доп. опции" ---
        self.check_temp_filter = QCheckBox("Временная медианная")
        self.check_temp_filter.stateChanged.connect(self.on_temp_filter_changed)
        self.spin_temp_window = QSpinBox()
        self.spin_temp_window.setRange(2, 10)
        self.spin_temp_window.setValue(3)
        self.spin_temp_window.valueChanged.connect(self.on_temp_window_changed)

        self.check_threshold = QCheckBox("Порог")
        self.check_threshold.stateChanged.connect(self.on_threshold_changed)
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(1, 255)
        self.spin_threshold.setValue(20)
        self.spin_threshold.valueChanged.connect(self.on_threshold_value_changed)

        self.check_adaptive_k = QCheckBox("Адаптивное k")
        self.check_adaptive_k.stateChanged.connect(self.on_adaptive_k_changed)
        self.spin_init_k = QDoubleSpinBox()
        self.spin_init_k.setRange(0.0, 2.0)
        self.spin_init_k.setSingleStep(0.01)
        self.spin_init_k.setValue(0.95)
        self.spin_init_k.valueChanged.connect(self.on_init_k_changed)
        self.spin_learn_rate = QDoubleSpinBox()
        self.spin_learn_rate.setRange(0.001, 0.1)
        self.spin_learn_rate.setSingleStep(0.001)
        self.spin_learn_rate.setValue(0.01)
        self.spin_learn_rate.valueChanged.connect(self.on_learn_rate_changed)

        opts_layout = QGridLayout()
        row = 0
        opts_layout.addWidget(self.check_temp_filter, row, 0)
        opts_layout.addWidget(self.spin_temp_window, row, 1)
        row += 1
        opts_layout.addWidget(self.check_threshold, row, 0)
        opts_layout.addWidget(self.spin_threshold, row, 1)
        row += 1
        opts_layout.addWidget(self.check_adaptive_k, row, 0)
        opts_layout.addWidget(self.spin_init_k, row, 1)
        opts_layout.addWidget(self.spin_learn_rate, row, 2)

        opts_group_box = QGroupBox("Доп. фильтры")
        opts_group_box.setLayout(opts_layout)

        # --- Выбор разрешения (ComboBox) ---
        self.combo_res = QComboBox()
        self.combo_res.currentIndexChanged.connect(self.on_resolution_changed)
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Разрешение:"))
        res_layout.addWidget(self.combo_res)
        res_group_box = QGroupBox("Настройки ROI / Размер")
        res_group_box.setLayout(res_layout)

        # --- Метки для отображения изображений ---
        self.label_orig = QLabel("Original")
        self.label_orig.setScaledContents(True)
        self.label_proc = QLabel("Processed")
        self.label_proc.setScaledContents(True)

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.label_orig)
        images_layout.addWidget(self.label_proc)

        # --- Кнопка Выход ---
        self.btn_exit = QPushButton("Выход")
        self.btn_exit.clicked.connect(self.close)

        # --- Общая компоновка ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(mode_group_box)
        main_layout.addWidget(opts_group_box)
        main_layout.addWidget(res_group_box)
        main_layout.addLayout(images_layout)
        main_layout.addWidget(self.btn_exit)

        self.setLayout(main_layout)
        self.setMinimumSize(900, 500)

    def init_camera(self):
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            raise mvsdk.CameraException(mvsdk.CAMERA_STATUS_NO_DEVICE_FOUND)

        i = 0  # ID камеры
        DevInfo = DevList[i]
        print("Use camera:", DevInfo.GetFriendlyName())

        # Открываем
        self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        self.capability = mvsdk.CameraGetCapability(self.hCamera)

        # Цвет/моно
        self.monoCamera = (self.capability.sIspCapacity.bMonoSensor != 0)
        if self.monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # Непрерывный режим
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # Выключаем автоэкспозицию, ставим 30 мс
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, 30_000)

        # Запуск потока
        mvsdk.CameraPlay(self.hCamera)

        # Выделяем буфер "по максимуму" (iWidthMax * iHeightMax * (1 или 3))
        max_w = self.capability.sResolutionRange.iWidthMax
        max_h = self.capability.sResolutionRange.iHeightMax
        FrameBufferSize = max_w * max_h
        if not self.monoCamera:
            FrameBufferSize *= 3
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        self.FrameBufferSize = FrameBufferSize

        count = self.capability.iImageSizeDesc
        pResDesc = self.capability.pImageSizeDesc
        self.resList = []
        self.combo_res.clear()

        for idx in range(count):
            # tSdkImageResolution
            res = pResDesc[idx]
            desc = res.GetDescription()    # строковое описание
            w = res.iWidthFOV
            h = res.iHeightFOV
            text = f"{desc} ({w}x{h})"
            self.resList.append(res.clone())  # Сохраним копию структуры
            self.combo_res.addItem(text)

        # Узнаем текущую настройку (CameraGetImageResolution)
        curRes = mvsdk.CameraGetImageResolution(self.hCamera)
        # Найдём в списке совпадение по iIndex или по (iWidthFOV,iHeightFOV)
        selected_index = 0
        for i, r in enumerate(self.resList):
            if r.iIndex == curRes.iIndex:
                selected_index = i
                break
            # Альтернативно, можно сравнивать iWidthFOV/iHeightFOV
            # if r.iWidthFOV == curRes.iWidthFOV and r.iHeightFOV == curRes.iHeightFOV:
            #     selected_index = i
            #     break

        self.combo_res.setCurrentIndex(selected_index)

        print("Camera init complete.")

    def on_resolution_changed(self, index):
        """
        Вызывается при смене пункта в ComboBox.
        Устанавливаем выбранное разрешение.
        """
        if self.hCamera is None:
            return
        if index < 0 or index >= len(self.resList):
            return

        # Берём соответствующую структуру
        newRes = self.resList[index]
        # Устанавливаем
        err = mvsdk.CameraSetImageResolution(self.hCamera, newRes)
        if err != 0:
            print("CameraSetImageResolution failed:", err)
            return

        print(f"Resolution changed: index={newRes.iIndex}, {newRes.iWidthFOV}x{newRes.iHeightFOV}")

        # Так как меняется геометрия кадра, очистим буферы
        self.frame_buffer.clear()
        self.diff_buffer.clear()

    #############################################
    # Обработка остальных UI-сигналов
    #############################################
    def on_mode_changed(self, mode_id):
        self.mode = mode_id
        self.frame_buffer.clear()
        self.diff_buffer.clear()

    def on_temp_filter_changed(self, state):
        self.use_temporal_filter = (state == Qt.Checked)
        if not self.use_temporal_filter:
            self.diff_buffer.clear()

    def on_temp_window_changed(self, value):
        self.temporal_window = value
        self.diff_buffer.clear()

    def on_threshold_changed(self, state):
        self.use_threshold = (state == Qt.Checked)

    def on_threshold_value_changed(self, value):
        self.threshold_value = value

    def on_adaptive_k_changed(self, state):
        self.use_adaptive_k = (state == Qt.Checked)
        self.adaptive_k.k = self.spin_init_k.value()

    def on_init_k_changed(self, value):
        self.base_k = value
        if not self.use_adaptive_k:
            self.adaptive_k.k = value

    def on_learn_rate_changed(self, value):
        self.adaptive_k.learning_rate = value

    #############################################
    # Захват и обработка кадров (таймер ~30 мс)
    #############################################
    def update_frame(self):
        if self.hCamera is None:
            return

        try:
            # Получаем RAW-данные
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            # Обрабатываем в pFrameBuffer (BGR8 или MONO8)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            # Освобождаем буфер
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            # Windows: переворот
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)

            # Формируем numpy
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)

            ch = 1 if (FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8) else 3
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, ch))

            # Показываем Original
            pix_orig = convert_frame_to_qpixmap(frame)
            self.label_orig.setPixmap(pix_orig)

            # Логика обработки
            self.frame_buffer.append(frame)
            current_k = self.adaptive_k.k if self.use_adaptive_k else self.base_k
            diff = None

            if self.mode == 1:
                # 30 Гц
                diff = process_mode_30(self.frame_buffer, current_k)
                if diff is not None:
                    # Оставляем 1 кадр
                    self.frame_buffer = self.frame_buffer[-1:]
            elif self.mode == 2:
                # 15 Гц
                diff = process_mode_15(self.frame_buffer, current_k)
                if diff is not None:
                    # Оставляем 2 кадра
                    self.frame_buffer = self.frame_buffer[-2:]
            elif self.mode == 3:
                # 20 Гц
                diff = process_mode_20(self.frame_buffer, current_k)
                if diff is not None:
                    # Полный сброс
                    self.frame_buffer.clear()

            if diff is not None:
                # Временная медианная
                if self.use_temporal_filter:
                    self.diff_buffer.append(diff)
                    if len(self.diff_buffer) > self.temporal_window:
                        self.diff_buffer.pop(0)
                    if len(self.diff_buffer) == self.temporal_window:
                        diff = temporal_median_filter(self.diff_buffer)
                else:
                    self.diff_buffer.clear()

                # Адаптивное k (обновим после вычисления diff)
                if self.use_adaptive_k:
                    # В зависимости от режима берём кадр "без подсветки"
                    # 30 Гц: последний кадр в frame_buffer
                    # 15 Гц: ...
                    # 20 Гц: ...
                    pass

                # Порог
                if self.use_threshold:
                    diff = apply_threshold(diff, self.threshold_value)

                # Показываем Processed
                pix_proc = convert_frame_to_qpixmap(diff)
                self.label_proc.setPixmap(pix_proc)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed:", e.message)

    def closeEvent(self, event):
        """
        При закрытии окна освобождаем камеру и буфер.
        """
        if self.hCamera is not None:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = None
        if self.pFrameBuffer is not None:
            mvsdk.CameraAlignFree(self.pFrameBuffer)
            self.pFrameBuffer = None
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
