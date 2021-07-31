from time import strftime, gmtime
from keras.preprocessing.image import img_to_array

def get_date():
    return strftime('%Y-%m-%d', gmtime())


# load and prepare the image
def load_image(img):
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    return img