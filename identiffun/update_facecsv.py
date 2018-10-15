# -*- utf-8 -*- 
# 

import os

f = open("./faces/face.csv", 'w')
fn = open("./faces.conf", 'w')
num = 0
namess = ""
for root, dirs, files in os.walk("./faces/"):
	for name in dirs:
		tmp_mu = os.path.join(root, name)
		namess+=(tmp_mu[8:] + ";")
		for root1, dirs1, files1 in os.walk(tmp_mu): 
			for name1 in files1:
				# print(os.path.join(root1, name1))
				f.write(os.path.join(root1, name1)[8:]+";%d\n" % num)
		num+=1
f.close()
fn.write(namess[:-1])
fn.close()

fm = open("./faces.conf", 'r')
print(fm.read())
fm.close()