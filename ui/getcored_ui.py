# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'getcored-1.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(947, 547)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(3, 3, 3, 3)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame = QtWidgets.QFrame(self.tab)
        self.frame.setMinimumSize(QtCore.QSize(0, 0))
        self.frame.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setContentsMargins(2, 2, 2, 2)
        self.gridLayout_2.setSpacing(1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.widget_2 = QtWidgets.QWidget(self.frame)
        self.widget_2.setObjectName("widget_2")
        self.gridLayout_2.addWidget(self.widget_2, 0, 1, 1, 1)
        self.gridLayout_3.addWidget(self.frame, 0, 0, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.tab)
        self.frame_2.setMinimumSize(QtCore.QSize(211, 0))
        self.frame_2.setMaximumSize(QtCore.QSize(211, 16777215))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_7.setSpacing(1)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.frame_6 = QtWidgets.QFrame(self.frame_2)
        self.frame_6.setMinimumSize(QtCore.QSize(211, 351))
        self.frame_6.setMaximumSize(QtCore.QSize(211, 351))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.widget = QtWidgets.QWidget(self.frame_6)
        self.widget.setGeometry(QtCore.QRect(10, 100, 183, 248))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.zhengshu_spinBox = QtWidgets.QSpinBox(self.widget)
        self.zhengshu_spinBox.setMinimumSize(QtCore.QSize(109, 27))
        self.zhengshu_spinBox.setMaximumSize(QtCore.QSize(27, 16777215))
        self.zhengshu_spinBox.setMaximum(1000000000)
        self.zhengshu_spinBox.setObjectName("zhengshu_spinBox")
        self.horizontalLayout_2.addWidget(self.zhengshu_spinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setMinimumSize(QtCore.QSize(64, 27))
        self.label_4.setMaximumSize(QtCore.QSize(64, 27))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_10.addWidget(self.label_4)
        self.fbl_comboBox = QtWidgets.QComboBox(self.widget)
        self.fbl_comboBox.setMinimumSize(QtCore.QSize(109, 27))
        self.fbl_comboBox.setMaximumSize(QtCore.QSize(109, 27))
        self.fbl_comboBox.setObjectName("fbl_comboBox")
        self.fbl_comboBox.addItem("")
        self.fbl_comboBox.addItem("")
        self.fbl_comboBox.addItem("")
        self.fbl_comboBox.addItem("")
        self.fbl_comboBox.addItem("")
        self.fbl_comboBox.addItem("")
        self.fbl_comboBox.addItem("")
        self.horizontalLayout_10.addWidget(self.fbl_comboBox)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_11 = QtWidgets.QLabel(self.widget)
        self.label_11.setMinimumSize(QtCore.QSize(64, 27))
        self.label_11.setMaximumSize(QtCore.QSize(64, 27))
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_8.addWidget(self.label_11)
        self.sd_SpinBox = QtWidgets.QDoubleSpinBox(self.widget)
        self.sd_SpinBox.setMinimumSize(QtCore.QSize(109, 27))
        self.sd_SpinBox.setMaximumSize(QtCore.QSize(109, 27))
        self.sd_SpinBox.setMaximum(10000000.0)
        self.sd_SpinBox.setSingleStep(0.1)
        self.sd_SpinBox.setObjectName("sd_SpinBox")
        self.horizontalLayout_8.addWidget(self.sd_SpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_12 = QtWidgets.QLabel(self.widget)
        self.label_12.setMinimumSize(QtCore.QSize(64, 27))
        self.label_12.setMaximumSize(QtCore.QSize(64, 27))
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_9.addWidget(self.label_12)
        self.bg_SpinBox = QtWidgets.QDoubleSpinBox(self.widget)
        self.bg_SpinBox.setMinimumSize(QtCore.QSize(109, 27))
        self.bg_SpinBox.setMaximumSize(QtCore.QSize(109, 27))
        self.bg_SpinBox.setMaximum(10000000.0)
        self.bg_SpinBox.setSingleStep(0.1)
        self.bg_SpinBox.setObjectName("bg_SpinBox")
        self.horizontalLayout_9.addWidget(self.bg_SpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setMinimumSize(QtCore.QSize(64, 27))
        self.label_5.setMaximumSize(QtCore.QSize(64, 27))
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_7.addWidget(self.label_5)
        self.ld_SpinBox = QtWidgets.QDoubleSpinBox(self.widget)
        self.ld_SpinBox.setMinimumSize(QtCore.QSize(109, 27))
        self.ld_SpinBox.setMaximumSize(QtCore.QSize(109, 27))
        self.ld_SpinBox.setMaximum(10000000.0)
        self.ld_SpinBox.setSingleStep(0.1)
        self.ld_SpinBox.setObjectName("ld_SpinBox")
        self.horizontalLayout_7.addWidget(self.ld_SpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setMinimumSize(QtCore.QSize(64, 27))
        self.label_6.setMaximumSize(QtCore.QSize(64, 27))
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.dbd_SpinBox = QtWidgets.QDoubleSpinBox(self.widget)
        self.dbd_SpinBox.setMinimumSize(QtCore.QSize(109, 27))
        self.dbd_SpinBox.setMaximumSize(QtCore.QSize(109, 27))
        self.dbd_SpinBox.setMaximum(10000000.0)
        self.dbd_SpinBox.setSingleStep(0.1)
        self.dbd_SpinBox.setObjectName("dbd_SpinBox")
        self.horizontalLayout_6.addWidget(self.dbd_SpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_13 = QtWidgets.QLabel(self.widget)
        self.label_13.setMinimumSize(QtCore.QSize(64, 27))
        self.label_13.setMaximumSize(QtCore.QSize(64, 27))
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_12.addWidget(self.label_13)
        self.zl_SpinBox = QtWidgets.QDoubleSpinBox(self.widget)
        self.zl_SpinBox.setMinimumSize(QtCore.QSize(109, 27))
        self.zl_SpinBox.setMaximumSize(QtCore.QSize(109, 27))
        self.zl_SpinBox.setMaximum(10000000.0)
        self.zl_SpinBox.setSingleStep(0.1)
        self.zl_SpinBox.setObjectName("zl_SpinBox")
        self.horizontalLayout_12.addWidget(self.zl_SpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setMinimumSize(QtCore.QSize(64, 27))
        self.label_7.setMaximumSize(QtCore.QSize(64, 27))
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_11.addWidget(self.label_7)
        self.bhd_SpinBox = QtWidgets.QDoubleSpinBox(self.widget)
        self.bhd_SpinBox.setMinimumSize(QtCore.QSize(109, 27))
        self.bhd_SpinBox.setMaximumSize(QtCore.QSize(109, 27))
        self.bhd_SpinBox.setMaximum(10000000.0)
        self.bhd_SpinBox.setSingleStep(0.1)
        self.bhd_SpinBox.setObjectName("bhd_SpinBox")
        self.horizontalLayout_11.addWidget(self.bhd_SpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_11)
        self.closeUSBBtn = QtWidgets.QPushButton(self.frame_6)
        self.closeUSBBtn.setGeometry(QtCore.QRect(10, 40, 91, 27))
        self.closeUSBBtn.setObjectName("closeUSBBtn")
        self.openFileBtn = QtWidgets.QPushButton(self.frame_6)
        self.openFileBtn.setGeometry(QtCore.QRect(110, 10, 91, 27))
        self.openFileBtn.setObjectName("openFileBtn")
        self.stopFileBtn = QtWidgets.QPushButton(self.frame_6)
        self.stopFileBtn.setGeometry(QtCore.QRect(160, 40, 41, 27))
        self.stopFileBtn.setObjectName("stopFileBtn")
        self.capBtn = QtWidgets.QPushButton(self.frame_6)
        self.capBtn.setGeometry(QtCore.QRect(60, 70, 41, 27))
        self.capBtn.setObjectName("capBtn")
        self.faceCheckBox = QtWidgets.QCheckBox(self.frame_6)
        self.faceCheckBox.setGeometry(QtCore.QRect(110, 70, 63, 29))
        self.faceCheckBox.setObjectName("faceCheckBox")
        self.openUSBBtn = QtWidgets.QPushButton(self.frame_6)
        self.openUSBBtn.setGeometry(QtCore.QRect(10, 10, 91, 27))
        self.openUSBBtn.setObjectName("openUSBBtn")
        self.startFileBtn = QtWidgets.QPushButton(self.frame_6)
        self.startFileBtn.setGeometry(QtCore.QRect(110, 40, 41, 27))
        self.startFileBtn.setObjectName("startFileBtn")
        self.luzhiBtn = QtWidgets.QPushButton(self.frame_6)
        self.luzhiBtn.setGeometry(QtCore.QRect(10, 70, 41, 27))
        self.luzhiBtn.setObjectName("luzhiBtn")
        self.gridLayout_7.addWidget(self.frame_6, 0, 0, 1, 1)
        self.info_textEdit = QtWidgets.QTextEdit(self.frame_2)
        self.info_textEdit.setObjectName("info_textEdit")
        self.gridLayout_7.addWidget(self.info_textEdit, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_2, 0, 1, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.frame_4 = QtWidgets.QFrame(self.tab_2)
        self.frame_4.setMinimumSize(QtCore.QSize(351, 0))
        self.frame_4.setMaximumSize(QtCore.QSize(351, 16777215))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.layoutWidget_2 = QtWidgets.QWidget(self.frame_4)
        self.layoutWidget_2.setGeometry(QtCore.QRect(10, 40, 337, 28))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget_2)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.sf_W_spinBox = QtWidgets.QSpinBox(self.layoutWidget_2)
        self.sf_W_spinBox.setMinimumSize(QtCore.QSize(81, 26))
        self.sf_W_spinBox.setMaximumSize(QtCore.QSize(81, 26))
        self.sf_W_spinBox.setMaximum(1000000000)
        self.sf_W_spinBox.setProperty("value", 100)
        self.sf_W_spinBox.setObjectName("sf_W_spinBox")
        self.horizontalLayout.addWidget(self.sf_W_spinBox)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget_2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.sf_H_spinBox = QtWidgets.QSpinBox(self.layoutWidget_2)
        self.sf_H_spinBox.setMinimumSize(QtCore.QSize(81, 26))
        self.sf_H_spinBox.setMaximumSize(QtCore.QSize(81, 26))
        self.sf_H_spinBox.setMaximum(1000000000)
        self.sf_H_spinBox.setProperty("value", 100)
        self.sf_H_spinBox.setObjectName("sf_H_spinBox")
        self.horizontalLayout.addWidget(self.sf_H_spinBox)
        self.layoutWidget = QtWidgets.QWidget(self.frame_4)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 70, 301, 28))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_8 = QtWidgets.QLabel(self.layoutWidget)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_4.addWidget(self.label_8)
        self.xz_X_spinBox = QtWidgets.QSpinBox(self.layoutWidget)
        self.xz_X_spinBox.setMinimumSize(QtCore.QSize(81, 26))
        self.xz_X_spinBox.setMaximumSize(QtCore.QSize(81, 26))
        self.xz_X_spinBox.setMaximum(1000000000)
        self.xz_X_spinBox.setProperty("value", 0)
        self.xz_X_spinBox.setObjectName("xz_X_spinBox")
        self.horizontalLayout_4.addWidget(self.xz_X_spinBox)
        self.label_10 = QtWidgets.QLabel(self.layoutWidget)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_4.addWidget(self.label_10)
        self.xz_Y_spinBox = QtWidgets.QSpinBox(self.layoutWidget)
        self.xz_Y_spinBox.setMinimumSize(QtCore.QSize(81, 26))
        self.xz_Y_spinBox.setMaximumSize(QtCore.QSize(81, 26))
        self.xz_Y_spinBox.setMaximum(1000000000)
        self.xz_Y_spinBox.setProperty("value", 0)
        self.xz_Y_spinBox.setObjectName("xz_Y_spinBox")
        self.horizontalLayout_4.addWidget(self.xz_Y_spinBox)
        self.layoutWidget1 = QtWidgets.QWidget(self.frame_4)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 100, 151, 28))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_9 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_3.addWidget(self.label_9)
        self.xz_D_spinBox = QtWidgets.QSpinBox(self.layoutWidget1)
        self.xz_D_spinBox.setMaximum(360)
        self.xz_D_spinBox.setObjectName("xz_D_spinBox")
        self.horizontalLayout_3.addWidget(self.xz_D_spinBox)
        self.frame_5 = QtWidgets.QFrame(self.frame_4)
        self.frame_5.setGeometry(QtCore.QRect(0, 170, 351, 251))
        self.frame_5.setMinimumSize(QtCore.QSize(351, 251))
        self.frame_5.setMaximumSize(QtCore.QSize(351, 251))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_5)
        self.gridLayout_6.setContentsMargins(2, 2, 2, 2)
        self.gridLayout_6.setSpacing(1)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.widget1 = QtWidgets.QWidget(self.frame_5)
        self.widget1.setObjectName("widget1")
        self.gridLayout_6.addWidget(self.widget1, 0, 0, 1, 1)
        self.openBtn = QtWidgets.QPushButton(self.frame_4)
        self.openBtn.setGeometry(QtCore.QRect(10, 10, 71, 26))
        self.openBtn.setMinimumSize(QtCore.QSize(71, 26))
        self.openBtn.setMaximumSize(QtCore.QSize(71, 26))
        self.openBtn.setObjectName("openBtn")
        self.pushFaceBtn = QtWidgets.QPushButton(self.frame_4)
        self.pushFaceBtn.setGeometry(QtCore.QRect(260, 460, 71, 25))
        self.pushFaceBtn.setObjectName("pushFaceBtn")
        self.cap_face_Btn = QtWidgets.QPushButton(self.frame_4)
        self.cap_face_Btn.setGeometry(QtCore.QRect(260, 430, 71, 25))
        self.cap_face_Btn.setObjectName("cap_face_Btn")
        self.widget2 = QtWidgets.QWidget(self.frame_4)
        self.widget2.setGeometry(QtCore.QRect(10, 130, 340, 33))
        self.widget2.setObjectName("widget2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget2)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.spBtn = QtWidgets.QPushButton(self.widget2)
        self.spBtn.setMinimumSize(QtCore.QSize(0, 26))
        self.spBtn.setMaximumSize(QtCore.QSize(16777215, 26))
        self.spBtn.setObjectName("spBtn")
        self.horizontalLayout_5.addWidget(self.spBtn)
        self.czBtn = QtWidgets.QPushButton(self.widget2)
        self.czBtn.setMinimumSize(QtCore.QSize(0, 26))
        self.czBtn.setMaximumSize(QtCore.QSize(16777215, 26))
        self.czBtn.setObjectName("czBtn")
        self.horizontalLayout_5.addWidget(self.czBtn)
        self.saveBtn = QtWidgets.QPushButton(self.widget2)
        self.saveBtn.setMinimumSize(QtCore.QSize(0, 26))
        self.saveBtn.setMaximumSize(QtCore.QSize(16777215, 26))
        self.saveBtn.setObjectName("saveBtn")
        self.horizontalLayout_5.addWidget(self.saveBtn)
        self.saveFaceBtn = QtWidgets.QPushButton(self.widget2)
        self.saveFaceBtn.setMinimumSize(QtCore.QSize(0, 26))
        self.saveFaceBtn.setMaximumSize(QtCore.QSize(16777215, 26))
        self.saveFaceBtn.setObjectName("saveFaceBtn")
        self.horizontalLayout_5.addWidget(self.saveFaceBtn)
        self.widget3 = QtWidgets.QWidget(self.frame_4)
        self.widget3.setGeometry(QtCore.QRect(180, 100, 161, 33))
        self.widget3.setObjectName("widget3")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.widget3)
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.face_1_checkBox = QtWidgets.QCheckBox(self.widget3)
        self.face_1_checkBox.setObjectName("face_1_checkBox")
        self.horizontalLayout_13.addWidget(self.face_1_checkBox)
        self.xzBtn = QtWidgets.QPushButton(self.widget3)
        self.xzBtn.setMinimumSize(QtCore.QSize(41, 26))
        self.xzBtn.setMaximumSize(QtCore.QSize(41, 26))
        self.xzBtn.setObjectName("xzBtn")
        self.horizontalLayout_13.addWidget(self.xzBtn)
        self.sfBtn = QtWidgets.QPushButton(self.widget3)
        self.sfBtn.setMinimumSize(QtCore.QSize(41, 26))
        self.sfBtn.setMaximumSize(QtCore.QSize(41, 26))
        self.sfBtn.setObjectName("sfBtn")
        self.horizontalLayout_13.addWidget(self.sfBtn)
        self.widget4 = QtWidgets.QWidget(self.frame_4)
        self.widget4.setGeometry(QtCore.QRect(10, 430, 241, 58))
        self.widget4.setObjectName("widget4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget4)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_14 = QtWidgets.QLabel(self.widget4)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_14.addWidget(self.label_14)
        self.ID_lineEdit = QtWidgets.QLineEdit(self.widget4)
        self.ID_lineEdit.setObjectName("ID_lineEdit")
        self.horizontalLayout_14.addWidget(self.ID_lineEdit)
        self.verticalLayout_2.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label_16 = QtWidgets.QLabel(self.widget4)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_15.addWidget(self.label_16)
        self.name_lineEdit = QtWidgets.QLineEdit(self.widget4)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.horizontalLayout_15.addWidget(self.name_lineEdit)
        self.verticalLayout_2.addLayout(self.horizontalLayout_15)
        self.gridLayout_4.addWidget(self.frame_4, 0, 0, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.tab_2)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.widget_3 = QtWidgets.QWidget(self.frame_3)
        self.widget_3.setMinimumSize(QtCore.QSize(0, 41))
        self.widget_3.setMaximumSize(QtCore.QSize(16777215, 41))
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_5.addWidget(self.widget_3, 0, 0, 1, 1)
        self.widget_4 = QtWidgets.QWidget(self.frame_3)
        self.widget_4.setObjectName("widget_4")
        self.gridLayout_5.addWidget(self.widget_4, 1, 0, 1, 1)
        self.gridLayout_4.addWidget(self.frame_3, 0, 1, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "GETTOOL"))
        self.label_2.setText(_translate("Form", "<html><head/><body><p align=\"right\">帧数：</p></body></html>"))
        self.label_4.setText(_translate("Form", "<html><head/><body><p align=\"right\">分辨率：</p></body></html>"))
        self.fbl_comboBox.setItemText(0, _translate("Form", "1920*1080"))
        self.fbl_comboBox.setItemText(1, _translate("Form", "1600*900"))
        self.fbl_comboBox.setItemText(2, _translate("Form", "1280*720"))
        self.fbl_comboBox.setItemText(3, _translate("Form", "800*600"))
        self.fbl_comboBox.setItemText(4, _translate("Form", "640*480"))
        self.fbl_comboBox.setItemText(5, _translate("Form", "600*400"))
        self.fbl_comboBox.setItemText(6, _translate("Form", "320*240"))
        self.label_11.setText(_translate("Form", "<html><head/><body><p align=\"right\">色调：</p></body></html>"))
        self.label_12.setText(_translate("Form", "<html><head/><body><p align=\"right\">曝光：</p></body></html>"))
        self.label_5.setText(_translate("Form", "<html><head/><body><p align=\"right\">亮度：</p></body></html>"))
        self.label_6.setText(_translate("Form", "<html><head/><body><p align=\"right\">对比度：</p></body></html>"))
        self.label_13.setText(_translate("Form", "<html><head/><body><p align=\"right\">帧率：</p></body></html>"))
        self.label_7.setText(_translate("Form", "<html><head/><body><p align=\"right\">饱和度：</p></body></html>"))
        self.closeUSBBtn.setText(_translate("Form", "关闭摄像头"))
        self.openFileBtn.setText(_translate("Form", "视频文件"))
        self.stopFileBtn.setText(_translate("Form", "暂停"))
        self.capBtn.setText(_translate("Form", "抓图"))
        self.faceCheckBox.setText(_translate("Form", "人脸"))
        self.openUSBBtn.setText(_translate("Form", "打开摄像头"))
        self.startFileBtn.setText(_translate("Form", "开始"))
        self.luzhiBtn.setText(_translate("Form", "录制"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "视频"))
        self.label.setText(_translate("Form", "<html><head/><body><p align=\"right\">缩放尺寸(像素W*H)：</p></body></html>"))
        self.label_3.setText(_translate("Form", "<html><head/><body><p align=\"center\">*</p></body></html>"))
        self.label_8.setText(_translate("Form", "<html><head/><body><p align=\"right\">旋转中心(X*Y)：</p></body></html>"))
        self.label_10.setText(_translate("Form", "<html><head/><body><p align=\"center\">*</p></body></html>"))
        self.label_9.setText(_translate("Form", "旋转角度："))
        self.openBtn.setText(_translate("Form", "打开文件"))
        self.pushFaceBtn.setText(_translate("Form", "导入人脸"))
        self.cap_face_Btn.setText(_translate("Form", "截取人脸"))
        self.spBtn.setText(_translate("Form", "水平镜像"))
        self.czBtn.setText(_translate("Form", "垂直镜像"))
        self.saveBtn.setText(_translate("Form", "保存图片"))
        self.saveFaceBtn.setText(_translate("Form", "保存人脸"))
        self.face_1_checkBox.setText(_translate("Form", "人脸"))
        self.xzBtn.setText(_translate("Form", "旋转"))
        self.sfBtn.setText(_translate("Form", "缩放"))
        self.label_14.setText(_translate("Form", "编号："))
        self.label_16.setText(_translate("Form", "姓名："))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "图片"))

