import pickle
from keras.preprocessing import image
import numpy as np


counter = 0


def prepare_dataset(no_imgs=-1):
    f_train_images = open('Automation.trainImages.txt', 'r')
    train_imgs = f_train_images.read().strip().split('\n') if no_imgs == - \
        1 else f_train_images.read().strip().split('\n')[:no_imgs]
    f_train_images.close()

    f_test_images = open('Automation.testImages.txt', 'r')
    test_imgs = f_test_images.read().strip().split('\n') if no_imgs == - \
        1 else f_test_images.read().strip().split('\n')[:no_imgs]
    f_test_images.close()

    f_train_dataset = open('temp/Automation_train_dataset.txt', 'w')
    f_train_dataset.write("image_id\tcaptions\n")

    f_test_dataset = open('temp/Automation_test_dataset.txt', 'w')
    f_test_dataset.write("image_id\tcaptions\n")

    f_captions = open('Automation.token.txt', 'r')
    captions = f_captions.read().strip().split('\n')
    data = {}
    for row in captions:
        row = row.split("\t")
        row[0] = row[0][:len(row[0]) - 2]
        try:
            data[row[0]].append(row[1])
        except:
            data[row[0]] = [row[1]]
    f_captions.close()


    c_train = 0
    for img in train_imgs:
        for capt in data[img]:
            caption = "<start> " + capt + " <end>"
            f_train_dataset.write(img + "\t" + caption + "\n")
            f_train_dataset.flush()
            c_train += 1
    f_train_dataset.close()

    c_test = 0
    for img in test_imgs:
        for capt in data[img]:
            caption = "<start> " + capt + " <end>"
            f_test_dataset.write(img + "\t" + caption + "\n")
            f_test_dataset.flush()
            c_test += 1
    f_test_dataset.close()
    return [c_train, c_test]

if __name__ == '__main__':
    c_train, c_test = prepare_dataset()
    print("Training samples = " + str(c_train) )
    print("Test samples = " + str(c_test))
