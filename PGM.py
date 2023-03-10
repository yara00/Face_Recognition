import cv2
import numpy as np


class PGM:
    def generate_nonface_imgs(self, sample_size):
        data_matrix = []
        label_matrix = []
        dir = ["car", "cat", "airplane", "fruit", "flower"]
        for i in range(len(dir)):
            for img_ctr in range(0, sample_size):
                if img_ctr < 10:
                    x = cv2.imread('data/natural_images/' + dir[i] + '/' + dir[i] + '_000' + str(img_ctr) + '.jpg',
                                   cv2.IMREAD_GRAYSCALE)
                elif img_ctr < 100:
                    x = cv2.imread('data/natural_images/' + dir[i] + '/' + dir[i] + '_00' + str(img_ctr) + '.jpg',
                                   cv2.IMREAD_GRAYSCALE)

                else:
                    x = cv2.imread('data/natural_images/' + dir[i] + '/' + dir[i] + '_0' + str(img_ctr) + '.jpg',
                                   cv2.IMREAD_GRAYSCALE)

                x = cv2.resize(x, (92, 112))
                image_vector = x.flatten()
                data_matrix.append(image_vector)
                label_matrix.append(0)
        return data_matrix, label_matrix


if __name__ == '__main__':
    mat, lab = PGM().generate_nonface_imgs()
    print(np.asarray(mat).shape)
    print(len(lab))
