import cv2
import numpy as np
import matplotlib.pyplot as plt

b = []
c = []
d = []
utk = []
row = 0
column = 0
count = 0
img = cv2.imread('tabaahi.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('q12.jpg', 0)
template2 = cv2.imread('q14.jpg', 0)
template3 = cv2.imread('asdf.png', 0)

w, h = template.shape[::-1]
res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc1 = np.where( res >= threshold)
for pt1 in zip(*loc1[::-1]):
	cv2.rectangle(img, pt1, (pt1[0] + w, pt1[1] + h), (0,255,255), 2)
	b.append(pt1)

w1, h1 = template2.shape[::-1]
res = cv2.matchTemplate(gray,template2,cv2.TM_CCOEFF_NORMED)
threshhold = 0.7
loc2 = np.where( res >= threshold)
for pt2 in zip(*loc2[::-1]):
	cv2.rectangle(img, pt2, (pt2[0] + w1, pt2[1] + h1), (100,255,0), 2)
	c.append(pt2)

w2, h2 = template3.shape[::-1] 
res = cv2.matchTemplate(gray,template3,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc3 = np.where( res >= threshold)
for pt3 in zip(*loc3[::-1]):
	cv2.rectangle(img, pt3, (pt3[0] + w2, pt3[1] + h2), (0,0,255), 2)
	d.append(pt3)

b.sort()
c.sort()
d.sort()

noofint = len (d)

for i in range (0, noofint):
	if d[0][0] == d[i][0]:
		row += 1
column = int (noofint/row + 1)
row += 1

height = d[1][1] - d[0][1]
width = d[row][0] - d[0][0]

total = 0
abtak = 0
a = []
a = [[0]*column for r in range (row)]

for i in range (0, len(b)):
	a[int(b[i][1] / height)][int(b[i][0] / width)] = -1
	total += 1

for i in range (0, len(c)):
	a[int(c[i][1] / height)][int(c[i][0] / width)] = 5000

i=0
j=0

def trace (l1, l2, l3, l4):
	q = [-l1,-l3]
	r = [l2,l4]
	plt.plot (r,q)

if a[0][0] == 5000:
	print("-1")
	quit()
if a[0][0] == -1:
	abtak +=1
	a[0][0] = 5000
if a[0][0] == 0:
	a[0][0] = 5000

def where (y, z, i, j):
	global total
	global abtak
	if i > 0 and a[i - 1][j] == -1:
		trace (i, j, i - 1, j)
		abtak += 1
		a[i - 1][j] = 5000
		where (i, j, i - 1, j)
	if i < (row - 1) and a[i + 1][j] == -1:
		trace (i, j, i + 1, j)
		abtak += 1
		a[i + 1][j] = 5000
		where (i, j, i + 1, j)
	if j < column - 1 and a[i][j + 1] == -1:
		trace (i, j, i, j + 1)
		abtak += 1
		a[i][j + 1] = 5000
		where (i, j, i, j + 1)
	if j > 0 and a[i][j - 1] == -1:
		trace (i, j, i, j - 1)
		abtak += 1
		a[i][j - 1] = 5000
		where (i, j, i, j - 1)
	if i < row - 1 and a[i + 1][j] == 0:
		a[i + 1][j] = 5000
		trace (i, j, i + 1, j)
		where (i, j, i + 1, j)
	if i > 0 and a[i - 1][j] == 0:
		a[i - 1][j] = 5000
		trace (i, j, i - 1, j)
		where (i, j, i - 1, j)
	if j < column - 1 and a[i][j + 1] == 0:
		a[i][j + 1] = 5000 
		trace (i, j, i, j + 1)
		where (i, j, i, j + 1)
	if j > 0 and a[i][j - 1] == 0:
		a[i][j - 1] = 5000
		trace (i, j, i, j - 1)
		where (i, j, i, j - 1)

if a[1][0] == -1:
	a[1][0] = 5000
	abtak+=1
	trace (0, 0, 1, 0)
	where (0, 0, 1, 0)
elif a[0][1] == -1:
	a[0][1] = 5000
	abtak+=1
	trace (0, 0, 0, 1)
	where (0, 0, 0, 1)
elif a[1][0] == 0:
	trace (0, 0, 1, 0)
	where (0, 0, 1, 0)
elif a[0][1] == 0:
	trace (0, 0, 0, 1)
	where (0, 0, 0, 1)

print ("total", total)
print ("Collected", abtak)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()