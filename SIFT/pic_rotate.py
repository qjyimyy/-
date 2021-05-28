from PIL import Image

img = Image.open(r".\corel\images\412.jpg")
rotate = img.rotate(15)
rotate.show()