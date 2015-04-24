from PIL import Image
import numpy as np
from image_ops import normalize_array


def image_as_array(image):
    assert image.mode in ['L', '1'], "Only Greyscale and Binary images supported"  # TODO: @motjuste Support Color Images?
    return np.asarray(image.getdata()).reshape((image.size[1], image.size[0]))


def import_image(from_location, as_array=False):
    try:
        image = Image.open(from_location)
    except IOError:
        raise IOError  # right now only checking IOError

    if as_array:
        return image_as_array(image)
    else:
        return image

def save_array_as_greyimage(array, saved_file, to_location="../generated/", normalize=False):
    assert len(array.shape) == 2, "only 2D arrays are accepted to generate a gray image"

    # normalize array data to range 0...255
    if normalize:
        array = normalize_array(array)

    # convert array to Image object
    result = Image.fromarray(array.astype(np.uint8), 'L')
    result.save(to_location + saved_file)


"""
#=========================TESTS=========================#
"""
import unittest

class TestImageIO(unittest.TestCase):

    from_location = "../resources/portrait.png"
    color_image = "../resources/TimFlach10.jpg"

    def test_import_image_IOError(self):
        with self.assertRaises(IOError):
            import_image(self.from_location[:-4], as_array=True)

    def test_import_image_ColorImageException(self):
        with self.assertRaises(AssertionError):
            import_image(self.color_image, as_array=True)

    def test_import_image_AsArray(self):
        self.assertIsInstance(import_image(self.from_location, as_array=True), np.ndarray)

    def test_import_image_AsImage(self):
        self.assertIsInstance(import_image(self.from_location, as_array=False), Image.Image)

    def test_save_array_as_greyimage_Normalized(self):
        saved_file = "portrait_normalized_test.png"
        save_array_as_greyimage(import_image(self.from_location, as_array=True), saved_file, normalize=True)
        print "Check saved file %s in generated folder" % saved_file
        # TODO: @motjuste is there a better way to test this?

    def test_save_array_as_greyimage_NotNormalized(self):
        saved_file = "portrait_notnormalized_test.png"
        save_array_as_greyimage(import_image(self.from_location, as_array=True), saved_file, normalize=False)
        print "Check saved file %s in generated folder" % saved_file
        # TODO: @motjuste is there a better way to test this?


if __name__ == "__main__":
    unittest.main()