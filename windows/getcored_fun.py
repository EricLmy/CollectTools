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
from identiffun.get_face import GenerateClass

class getDataWindows(QWidget):
    def __del__(self):
        if hasattr(self, "camera"):
            self.camera.release()# 释放资源

    def init_fun(self):
        self.window = Ui_Form()
        self.window.setupUi(self)

        # self.timer = QTimer()# 定义一个定时器对象
        # self.timer.timeout.connect(self.timer_fun) #计时结束调用方法

        # self.window.openBtn.clicked.connect(self.timer_start)
        # self.window.closeBtn.clicked.connect(self.closeBtn_fun)

        # self.window.saveBtn.clicked.connect(self.saveBtn_fun)

        # self.window.z1.clicked.connect(self.frontal_face_z_fun)
        # self.window.z2.clicked.connect(self.frontal_face_z_fun)
        # self.window.z3.clicked.connect(self.frontal_face_z_fun)
        # self.window.z4.clicked.connect(self.frontal_face_z_fun)
        # self.window.z5.clicked.connect(self.frontal_face_z_fun)

        # self.window.s1.clicked.connect(self.frontal_face_z_fun)
        # self.window.s2.clicked.connect(self.frontal_face_z_fun)
        # self.window.s3.clicked.connect(self.frontal_face_z_fun)
        # self.window.s4.clicked.connect(self.frontal_face_z_fun)
        # self.window.s5.clicked.connect(self.frontal_face_z_fun)
        # self.window.x1.clicked.connect(self.frontal_face_z_fun)
        # self.window.x2.clicked.connect(self.frontal_face_z_fun)
        # self.window.x3.clicked.connect(self.frontal_face_z_fun)
        # self.window.x4.clicked.connect(self.frontal_face_z_fun)
        # self.window.x5.clicked.connect(self.frontal_face_z_fun)

        # self.window.zc1.clicked.connect(self.frontal_face_zc_fun)
        # self.window.zc2.clicked.connect(self.frontal_face_zc_fun)
        # self.window.zc3.clicked.connect(self.frontal_face_zc_fun)
        # self.window.zc4.clicked.connect(self.frontal_face_zc_fun)
        # self.window.zc5.clicked.connect(self.frontal_face_zc_fun)

        # self.window.yc1.clicked.connect(self.frontal_face_yc_fun)
        # self.window.yc2.clicked.connect(self.frontal_face_yc_fun)
        # self.window.yc3.clicked.connect(self.frontal_face_yc_fun)
        # self.window.yc4.clicked.connect(self.frontal_face_yc_fun)
        # self.window.yc5.clicked.connect(self.frontal_face_yc_fun)

        # self.profile_face_flag = True # true左侧脸，false:右侧脸

        # self.my_get_face = GenerateClass()

        # self.face_data = {}        
        # self.face_data = {"z1":None,"z2":None,"z3":None,"z4":None,"z5":None,
        #                   "s1":None,"s2":None,"s3":None,"s4":None,"s5":None,
        #                   "x1":None,"x2":None,"x3":None,"x4":None,"x5":None,
        #                   "yc1":None,"yc2":None,"yc3":None,"yc4":None,"yc5":None,
        #                   "zc1":None,"zc2":None,"zc3":None,"zc4":None,"zc5":None}

    # def frontal_face_yc_fun(self):
    #     if hasattr(self, "camera"):
    #         self.profile_face_flag = False
    #         button = self.sender()
    #         self.my_get_face.profile_face() # you ce lian
    #         face_img = self.my_get_face.get_gray_data()
    #         self.face_data[button.objectName()] = face_img
    #         cv2.imwrite("./image/btn_pg.jpg", cv2.resize(cv2.flip(face_img.copy(), 1, dst=None), (71, 71)))

    #         button.setIconSize(QSize(71,71))
    #         button.setStyleSheet("background-image:url(./image/btn_pg.jpg);")

    # def frontal_face_zc_fun(self):
    #     if hasattr(self, "camera"):
    #         self.profile_face_flag = True
    #         button = self.sender()
    #         self.my_get_face.profile_face() # zuo ce lian
    #         face_img = self.my_get_face.get_gray_data()
    #         self.face_data[button.objectName()] = face_img
    #         cv2.imwrite("./image/btn_pg.jpg", cv2.resize(face_img.copy(), (71, 71)))

    #         button.setIconSize(QSize(71,71))
    #         button.setStyleSheet("background-image:url(./image/btn_pg.jpg);")

    # def frontal_face_z_fun(self):
    #     if hasattr(self, "camera"):
    #         self.profile_face_flag = True
    #         button = self.sender()
    #         self.my_get_face.frontal_face() # zheng lian
    #         face_img = self.my_get_face.get_gray_data()
    #         self.face_data[button.objectName()] = face_img
    #         cv2.imwrite("./image/btn_pg.jpg", cv2.resize(face_img.copy(), (71, 71)))

    #         button.setIconSize(QSize(71,71))
    #         button.setStyleSheet("background-image:url(./image/btn_pg.jpg);")

    # def saveBtn_fun(self):
    #     name = self.window.name_lineEdit.text()
    #     # print(name)
    #     rootdir = os.path.abspath('.')
    #     # print(rootdir)
    #     filedir = "identiffun/faces/%s/" % name 
    #     facesdir = os.path.join(rootdir, filedir)
    #     if os.path.exists(facesdir):
    #         pass
    #         # #cunzai error
    #         # print("[error]exists this files")
    #         # return
    #     else:
    #         os.mkdir(facesdir)
    #     # # 1. get info 
    #     # filedir = "./identiffun/faces/%s/" % name 
    #     for key in self.face_data:
    #         cv2.imwrite('%s/%s.pgm' % (facesdir,key), self.face_data[key]) 

    # def closeBtn_fun(self):
    #     if hasattr(self, "camera"):
    #         self.camera.release()# 释放资源
    #         self.timer.stop()
    #     self.window.figaxes.clear()
    #     self.window.figure.canvas.draw()


    # def timer_fun(self):
    #     ret, frame = self.camera.read()
    #     if ret:
    #         self.showimg2figaxes(frame)
    #     else:
    #         self.timer.stop()

    # def timer_start(self):
    #     if hasattr(self, "camera"):
    #         if not self.camera.isOpened():
    #             self.camera.open(0)
    #     else:
    #         self.camera = cv2.VideoCapture(0)

    #     if self.camera.isOpened():
    #         pass
    #     else:
    #         print("not Open USB")
    #         return
        
    #     self.timer.start(100) #设置计时间隔并启动


    # def showimg2figaxes(self,img):
    #     if self.profile_face_flag:
    #         tmp_img = self.my_get_face.get_face_fun(img)
    #     else:
    #         tmp_img = self.my_get_face.get_face_fun(cv2.flip(img, 1, dst=None))
    #         tmp_img = cv2.flip(tmp_img, 1, dst=None)
    #     b, g, r = cv2.split(tmp_img)
    #     imgret = cv2.merge([r,g,b])# 这个就是前面说书的，OpenCV和matplotlib显示不一样，需要转换
    #     self.window.figaxes.clear()
    #     self.window.figaxes.imshow(imgret)
    #     self.window.figure.canvas.draw()


if __name__ == '__main__':
    
    import sys
    app = QApplication(sys.argv)
    mainW = QMainWindow()
    ui = getDataWindows(mainW)
    ui.init_fun()
    mainW.show()
    sys.exit(app.exec_())