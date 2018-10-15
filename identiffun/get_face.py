import cv2 
import numpy as np 
import sys 
import os

fileconf = "/home/lmy/gRPC/grpcdemo/tools/identiffun/faces.conf"
filedata = "/home/lmy/gRPC/grpcdemo/tools/identiffun/faces/"
filexml = "/home/lmy/gRPC/grpcdemo/tools/identiffun/haarcascades/haarcascade_frontalface_default.xml"
filexml_z1 = "/home/lmy/gRPC/grpcdemo/tools/identiffun/haarcascades/haarcascade_frontalface_alt.xml"
filexml_z2 = "/home/lmy/gRPC/grpcdemo/tools/identiffun/haarcascades/haarcascade_frontalface_alt2.xml"
filexmlzc = "/home/lmy/gRPC/grpcdemo/tools/identiffun/haarcascades/haarcascade_profileface.xml"

filexml_cat = "/home/lmy/gRPC/grpcdemo/tools/identiffun/haarcascades/haarcascade_frontalcatface.xml" # cat fance
filexml1 = "/home/lmy/gRPC/grpcdemo/tools/identiffun/haarcascades/haarcascade_frontalface_alt_tree.xml"

class GenerateClass(object):
	"""docstring for GenerateClass"""
	def __init__(self):
		super(GenerateClass, self).__init__()

		self.face_cascade = cv2.CascadeClassifier(filexml_z1)
		
	def frontal_face(self):
		self.face_cascade = cv2.CascadeClassifier(filexml_z1)

	def profile_face(self):
		self.face_cascade = cv2.CascadeClassifier(filexmlzc)

	def get_gray_data(self):
		if hasattr(self, "ret_f"):
			return self.ret_f

	def get_face_fun(self, img):
		self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
		for (x,y,w,h) in self.faces:
			ret_img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 1)
			self.ret_f = cv2.resize(self.gray[y:y+h, x:x+w], (200, 200))

		return img
		# return img, ret_f


def read_images(path, sz=None):
	c = 0
	X, y = [], []
	for dirname, dirnames, filenames in os.walk(path):
		for subdirname in dirnames:
			subject_path = os.path.join(dirname, subdirname)
			for filename in os.listdir(subject_path):
				try:
					if (filename == '.directory'):
						continue
					filepath = os.path.join(subject_path, filename)
					im = cv2.imread(os.path.join(subject_path, filename),cv2.IMREAD_GRAYSCALE)
					# resize to given size (if given)
					if (sz is not None):
						im = cv2.resize(im, (200, 200))
					X.append(im)
					y.append(c)
				except:
					print('Unexpected error:', sys.exc_info()[0])
					raise 
			c = c+1
	return [X, y]

class Get_Faces(object):
	"""docstring for Get_Faces"""
	def __init__(self, names):
		super(Get_Faces, self).__init__()
		self.names1 = names
		[self.X, self.y] = read_images(filedata)
		self.y = np.asarray(self.y, dtype=np.int32)
		
		self.model = cv2.face.EigenFaceRecognizer_create()#cv2.face.createEigenFaceRecognizer()
		# self.model = cv2.face.FisherFaceRecognizer_create()#cv2.face.createFisherFaceRecognizer()
		# self.model = cv2.face.LBPHFaceRecognizer_create()#cv2.face.createLBPHFaceRecognizer()
		self.model.train(np.asarray(self.X), np.asarray(self.y))
		self.face_cascade = cv2.CascadeClassifier(filexml_z1)
		self.face_cascade1 = cv2.CascadeClassifier(filexmlzc)
		# print(self.names)

	def get_face_fun(self, img):#正脸
		faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
		faces1 = self.face_cascade1.detectMultiScale(img, 1.3, 5)
		faces2 = self.face_cascade1.detectMultiScale(cv2.flip(img, 1, dst=None), 1.3, 5)
		for (x,y,w,h) in faces:
			img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			roi = gray[x:x+w, y:y+h]
			try:
				roi = cv2.resize(roi, (200,200), interpolation=cv2.INTER_LINEAR)
				params = self.model.predict(roi)
				print("Label:%s, Confidence:%.2f" % (params[0], params[1]))# params[1]:ke xin du
				# if params[1] < 1100.0:
				cv2.putText(img, self.names1[params[0]], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
			except:
				continue

		for (x,y,w,h) in faces1:# 左侧脸
			img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			roi = gray[x:x+w, y:y+h]
			try:
				roi = cv2.resize(roi, (200,200), interpolation=cv2.INTER_LINEAR)
				params = self.model.predict(roi)
				print("Label:%s, Confidence:%.2f" % (params[0], params[1]))# params[1]:ke xin du
				# if params[1] < 1100.0:
				cv2.putText(img, self.names1[params[0]], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
			except:
				continue

		for (x,y,w,h) in faces2: # 右侧脸
			x = img.shape[1] - x - w
			img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			roi = gray[x:x+w, y:y+h]
			try:
				roi = cv2.resize(roi, (200,200), interpolation=cv2.INTER_LINEAR)
				params = self.model.predict(roi)
				print("Label:%s, Confidence:%.2f" % (params[0], params[1]))# params[1]:ke xin du
				# if params[1] < 1100.0:
				cv2.putText(img, self.names1[params[0]], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
			except:
				continue

		return img


