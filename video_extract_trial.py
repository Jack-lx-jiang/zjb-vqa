import imageio

base_dir = 'dataset'
filename = base_dir + '/train/' + 'ZJL1.mp4'
vid = imageio.get_reader(filename, 'ffmpeg')
print(vid.get_length())
# print(vid.format)
im = vid.get_data(vid.get_length() - 2)
# print(im.shape)
# im = transform.resize(im,(224,224))
# print(im.shape)
# im1 = vid.get_next_data()
# im2 = vid.get_next_data()
# while(True):
#     im2 = vid.get_next_data()
# print(im1==im2)


# image = skimage.img_as_float(im).astype(np.float64)
# print(image.size)
# print(type(image))

# vid = VideoCapture(base_dir)
# vid.open()
#
# retval, image = vid.read()
# print(image.size)
# print(type(image))
