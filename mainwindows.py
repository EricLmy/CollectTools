# -*- coding: utf-8 -*-

from PyQt5.QtCore import QDate, QFile, Qt, QTextStream, QSize,QPoint,QSettings
from PyQt5.QtGui import (QFont, QIcon, QKeySequence, QTextCharFormat,
        QTextCursor, QTextTableFormat)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import (QAction, QApplication, QDialog, QDockWidget,QStackedWidget,
        QFileDialog, QListWidget, QMainWindow, QMessageBox, QTextEdit)

# import dockwidgets_rc

from windows.getcored_fun import getDataWindows

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.stackedWidget = QStackedWidget()
        self.setCentralWidget(self.stackedWidget)

        self.createActions()
        self.createMenus()
        self.createToolBars()
        self.createStatusBar()

        self.createMyWindows()

        self.setWindowTitle("Tools")

        self.readSettings()

    def closeEvent(self, event):
        self.writeSettings()
        event.accept()

    def createMyWindows(self):
        self.getDataW = getDataWindows()
        self.getDataW.init_fun()
        self.stackedWidget.addWidget(self.getDataW)


    # def usb1w(self):
    #     self.stackedWidget.setCurrentIndex(0)

    # def demo1w(self):
    #     self.stackedWidget.setCurrentIndex(2)

    def getDataW(self):
        self.stackedWidget.setCurrentIndex(0)

    def about(self):
        QMessageBox.about(self, "About Dock Widgets",
                "The <b>Dock Widgets</b> example demonstrates how to use "
                "Qt's dock widgets. You can enter your own text, click a "
                "customer to add a customer name and address, and click "
                "standard paragraphs to add them.")

    def createActions(self):
        self.getDataAct = QAction(QIcon('./image/xiaob.jpg'), "demo2", self,
                statusTip="get data windows", triggered=self.getDataW)

        self.quitAct = QAction("&Quit", self, shortcut="Ctrl+Q",
                statusTip="Quit the application", triggered=self.close)

        self.aboutAct = QAction("&About", self,
                statusTip="Show the application's About box",
                triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                statusTip="Show the Qt library's About box",
                triggered=QApplication.instance().aboutQt)

    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu("摄像头操作")
        self.fileMenu.addAction(self.getDataAct)
        # self.fileMenu.addSeparator()
        # self.fileMenu.addAction(self.quitAct)

        self.editMenu = self.menuBar().addMenu("图片预览")
        # self.editMenu.addAction(self.undoAct)

        self.viewMenu = self.menuBar().addMenu("录像操作")

        self.menuBar().addSeparator()

        self.helpMenu = self.menuBar().addMenu("帮助")
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.quitMenu = self.menuBar().addMenu("退出")
        self.quitMenu.addAction(self.quitAct)

    def createToolBars(self):
        self.fileToolBar = self.addToolBar("File")
        self.fileToolBar.addAction(self.getDataAct)

        self.editToolBar = self.addToolBar("Edit")
        # self.editToolBar.addAction(self.undoAct)

    def createStatusBar(self):
        self.statusBar().showMessage("Ready")

    def readSettings(self):
        settings = QSettings('Trolltech', 'MDI Example')
        pos = settings.value('pos', QPoint(100, 100))
        size = settings.value('size', QSize(914, 589))
        self.move(pos)
        self.resize(size)

    def writeSettings(self):
        settings = QSettings('Trolltech', 'MDI Example')
        settings.setValue('pos', self.pos())
        settings.setValue('size', self.size())

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
