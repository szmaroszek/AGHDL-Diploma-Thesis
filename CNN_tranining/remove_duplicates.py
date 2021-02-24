from imutils import paths
import cv2
import os

def dhash(image, hashSize=8):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))

	diff = resized[:, 1:] > resized[:, :-1]
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

imagePaths = list(paths.list_images("###"))
hashes = {}

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	h = dhash(image)
	p = hashes.get(h, [])
	p.append(imagePath)
	hashes[h] = p

for (h, hashedPaths) in hashes.items():
    if len(hashedPaths) > 1:
        for p in hashedPaths[1:]:
            os.remove(p)