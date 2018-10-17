# -*- coding:utf-8 -*- 
# 

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QTimer, QSize
if __name__ == '__main__':
    from getcored_ui import Ui_Form
else:
    from windows.getcored_ui import Ui_Form

import cv2 
import os
import copy
from identiffun.get_face import GenerateClass

class getDataWindows(QWidget):
    def __del__(self):
        if hasattr(self, "camera"):
            self.camera.release()# 释放资源

    def init_fun(self):
        self.window = Ui_Form()
        self.window.setupUi(self)

        self.timer = QTimer()# 定义一个定时器对象
        self.timer.timeout.connect(self.timer_fun) #计时结束调用方法

        self.window.openUSBBtn.clicked.connect(self.timer_start)
        self.window.closeUSBBtn.clicked.connect(self.closeBtn_fun)

        self.window.fbl_comboBox.currentIndexChanged.connect(self.set_width_and_height)

        self.window.capBtn.clicked.connect(self.catch_picture)
        self.window.saveBtn.clicked.connect(self.saveBtn_fun)

    def catch_picture(self):
        if hasattr(self, "camera") and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.raw_frame = copy.deepcopy(frame)
                self.showimg2picfigaxes(frame)
            else:
                pass # get faild

    def saveBtn_fun(self):
        filename, filetype = QFileDialog.getSaveFileName(self, "save", "", "jpg Files(*.jpg)::All Files(*)")
        if filename:
            cv2.imwrite(filename, self.raw_frame)

    def set_width_and_height(self):
        width, height = self.window.fbl_comboBox.currentText().split('*')
        if hasattr(self, "camera"):
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def closeBtn_fun(self):
        if hasattr(self, "camera"):
            self.camera.release()# 释放资源
            self.timer.stop()
        self.window.figaxes_video.clear()
        self.window.figure_video.canvas.draw()


    def timer_fun(self):
        ret, frame = self.camera.read()
        if ret:
            self.showimg2videofigaxes(frame)
        else:
            self.timer.stop()

    def timer_start(self):
        if hasattr(self, "camera"):
            if not self.camera.isOpened():
                self.camera.open(0)
        else:
            self.camera = cv2.VideoCapture(0)

        if self.camera.isOpened():
            pass
        else:
            print("not Open USB")
            return
        self.get_camera_params()
        self.timer.start(41) #设置计时间隔并启动

    def get_camera_params(self):
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.window.fbl_comboBox.setCurrentText("%d*%d" % (int(width), int(height)))

        fps = self.camera.get(cv2.CAP_PROP_FPS)
        if fps == float('inf'):
            self.window.zl_SpinBox.setValue(0.0)
        else:
            self.window.zl_SpinBox.setValue(fps)

        brightness = self.camera.get(cv2.CAP_PROP_BRIGHTNESS)
        if brightness == float('inf'):
            self.window.ld_SpinBox.setValue(0.0)
        else:
            self.window.ld_SpinBox.setValue(brightness)

        contrast = self.camera.get(cv2.CAP_PROP_CONTRAST)
        if contrast == float('inf'):
            self.window.dbd_SpinBox.setValue(0.0)
        else:
            self.window.dbd_SpinBox.setValue(contrast)

        hue = self.camera.get(cv2.CAP_PROP_HUE)
        if hue == float('inf'):
            self.window.sd_SpinBox.setValue(0.0)
        else:
            self.window.sd_SpinBox.setValue(hue)

        exposure =self.camera.get(cv2.CAP_PROP_EXPOSURE) 
        if exposure == float('inf'):
            self.window.bg_SpinBox.setValue(0.0)
        else:
            self.window.bg_SpinBox.setValue(exposure) # inf

        saturation =self.camera.get(cv2.CAP_PROP_SATURATION) 
        if saturation == float('inf'):
            self.window.bhd_SpinBox.setValue(0.0)
        else:
            self.window.bhd_SpinBox.setValue(saturation) # inf

    def showimg2videofigaxes(self, img):
        b, g, r = cv2.split(img)
        imgret = cv2.merge([r,g,b])# 这个就是前面说书的，OpenCV和matplotlib显示不一样，需要转换
        self.window.figaxes_video.clear()
        self.window.figaxes_video.imshow(imgret)
        self.window.figure_video.canvas.draw()


    def showimg2picfigaxes(self,img):
        b, g, r = cv2.split(img)
        imgret = cv2.merge([r,g,b])# 这个就是前面说书的，OpenCV和matplotlib显示不一样，需要转换
        self.window.figaxes_pic.clear()
        self.window.figaxes_pic.imshow(imgret)
        self.window.figure_pic.canvas.draw()


if __name__ == '__main__':
    
    import sys
    app = QApplication(sys.argv)
    mainW = QMainWindow()
    ui = getDataWindows(mainW)
    ui.init_fun()
    mainW.show()
    sys.exit(app.exec_())