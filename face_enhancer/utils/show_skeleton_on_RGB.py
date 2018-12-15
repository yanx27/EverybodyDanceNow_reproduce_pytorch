import imageio
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

testVideo = 'D:/data/ntu_processed_256/S001C001P001R001A001_rgb.avi'
testSkeleton = 'D:/data/ntu_processed_256/S001C001P001R001A001.skeleton'
outputVideo = 'D:/data/out_256.avi'

def render_frame(frame, skeleton):
	connecting_joint = np.array(
		[2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]) - 1

	for i in range(25):
		dx = int(skeleton[i, 0])
		dy = int(skeleton[i, 1])

		# if skeletons[n][i, 2] == 0:
		#	continue

		rv = 255
		gv = 0
		bv = 0

		k = connecting_joint[i]

		dx2 = int(skeleton[k, 0])
		dy2 = int(skeleton[k, 1])

		xdist = abs(dx - dx2)
		ydist = abs(dy - dy2)

		if xdist > ydist:
			xrange = np.linspace(dx, dx2, xdist, endpoint=False)
			yrange = np.linspace(dy, dy2, xdist, endpoint=False)
		else:
			yrange = np.linspace(dy, dy2, ydist, endpoint=False)
			xrange = np.linspace(dx, dx2, ydist, endpoint=False)

		for i in range(len(xrange)):
			dx = int(round(xrange[i]))
			dy = int(round(yrange[i]))
			frame[dy - 1: dy + 1, dx - 1: dx + 1, 0] = rv
			frame[dy - 1: dy + 1, dx - 1: dx + 1, 1] = gv
			frame[dy - 1: dy + 1, dx - 1: dx + 1, 2] = bv

		rv = 0
		gv = 255
		bv = 0
		frame[dy - 2: dy + 2, dx - 2: dx + 2, 0] = rv
		frame[dy - 2: dy + 2, dx - 2: dx + 2, 1] = gv
		frame[dy - 2: dy + 2, dx - 2: dx + 2, 2] = bv

	return frame


if __name__ == '__main__':
	videoreader = imageio.get_reader(testVideo)
	fps = videoreader._meta['fps']
	frames = [im for im in videoreader]
	skeletons = loadmat(testSkeleton)
	skeletons = skeletons['joint_coordinates']
	assert (len(frames) == skeletons.shape[0])

	videowriter = imageio.get_writer(outputVideo, fps=fps)

	for n in range(len(frames)):
		videowriter.append_data(render_frame(frames[n], skeletons[n, ...]))

	videowriter.close()
