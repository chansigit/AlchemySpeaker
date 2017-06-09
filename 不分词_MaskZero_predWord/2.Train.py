import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence
from keras.callbacks import ModelCheckpoint
import pickle
import sys

EMBEDDING_DIM = 128


class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.f_train_dataset_name = 'PR_train_dataset.txt'
        self.f_test_dataset_name = 'PR_test_dataset.txt'
        self.encoded_images = pickle.load(open("encoded_images.p", "rb"))
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv(self.f_train_dataset_name, delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = next(iter)
            caps.append(x[1][1])

        self.total_samples = 0
        for text in caps:
            self.total_samples += len(text.split()) - 1
        print("Total samples : " + str(self.total_samples))

        # 在数过了训练样本的个数后，应该补上测试样本，来扩充词表
        # 不过这时候就不影响total_samples了，total_samples仍然是训练样本个数。
        test_df = pd.read_csv(self.f_test_dataset_name, delimiter='\t')
        test_nb_samples = test_df.shape[0]

        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}

        # 对词语编号应该从1开始，而不是0
        # 这是因为0要留给caption向量作padding zero用
        # 所以下面word对应的位置是i+1，而不是i
        for i, word in enumerate(unique):
            self.word_index[word] = i
            self.index_word[i] = word

        # 更新caption值
        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print("Vocabulary size: " + str(self.vocab_size))
        print("Maximum caption length: " + str(self.max_cap_len))
        print("Variables initialization done!")

    def train_data_generator(self, batch_size=32):
        partial_caps = []
        next_words = []
        images = []
        print("Generating data...")
        gen_count = 0
        df = pd.read_csv(self.f_train_dataset_name, delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = next(iter)
            caps.append(x[1][1])
            imgs.append(x[1][0])

        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter += 1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split()) - 1):
                    total_count += 1

                    #partial = [self.word_index[txt] for txt in text.split()[:i+1]]

                    partial = []
                    for txt in text.split()[:i + 1]:
                        try:
                            partial.append(self.word_index[txt])
                        except KeyError:  # trainset里面找不到 txt 这个字
                            # 那就把这个字打印到屏幕上去
                            print("Info (train-txt): trainset has no ", txt)
                            # 然后再用一个trainset里面有的'某'字代替
                            partial.append(self.word_index['某'])

                    partial_caps.append(partial)
                    nextItem = np.zeros(self.vocab_size)
                    try:
                        # trainset里面找不到 text.split()[i+1] 这个字
                        nextItem[self.word_index[text.split()[i + 1]]] = 1
                    except KeyError:
                        # 那就把这个字打印到屏幕上去
                        print("Info (train-split): trainset has no ",
                              text.split()[i + 1])
                        # 然后再用一个trainset里面有的'某'字代替
                        nextItem[self.word_index['某']] = 1
                    next_words.append(nextItem)
                    images.append(current_image)

                    if total_count >= batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(
                            partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count += 1
                        if gen_count % 50 == 0:
                            print("yielding count: " + str(gen_count))
                            sys.stdout.flush()

                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []

    def test_data_generator(self, batch_size=32):
        partial_caps = []
        next_words = []
        images = []
        print("Generating data...")
        gen_count = 0
        df = pd.read_csv(self.f_test_dataset_name, delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = next(iter)
            caps.append(x[1][1])
            imgs.append(x[1][0])

        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter += 1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split()) - 1):
                    total_count += 1
                    #partial = [self.word_index[txt] for txt in text.split()[:i+1]]

                    partial = []
                    for txt in text.split()[:i + 1]:
                        try:
                            partial.append(self.word_index[txt])
                        except KeyError:  # trainset里面找不到 txt 这个字
                            # 那就把这个字打印到屏幕上去
                            print("Info (test-txt): trainset has no ", txt)
                            # 然后再用一个trainset里面有的'某'字代替
                            partial.append(self.word_index['某'])

                    partial_caps.append(partial)
                    nextItem = np.zeros(self.vocab_size)

                    try:
                        # trainset里面找不到 text.split()[i+1] 这个字
                        nextItem[self.word_index[text.split()[i + 1]]] = 1
                    except KeyError:
                        # 那就把这个字打印到屏幕上去
                        print("Info (test-split): trainset has no ",
                              text.split()[i + 1])
                        # 然后再用一个trainset里面有的'某'字代替
                        nextItem[self.word_index['某']] = 1

                    next_words.append(nextItem)
                    images.append(current_image)

                    if total_count >= batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(
                            partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count += 1
                        if gen_count % 50 == 0:
                            print("yielding count: " + str(gen_count))
                            sys.stdout.flush()

                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []

    def load_image(self, path):
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.asarray(x)

    def create_model(self, ret_model=False):
        #base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
        # base_model.trainable=False
        image_model = Sequential()
        # image_model.add(base_model)
        # image_model.add(Flatten())
        image_model.add(
            Dense(EMBEDDING_DIM, input_dim=4096, activation='relu'))

        image_model.add(RepeatVector(self.max_cap_len))

        lang_model = Sequential()
        lang_model.add(Embedding(self.vocab_size, 256,
                                 input_length=self.max_cap_len))
        lang_model.add(LSTM(256, return_sequences=True))
        lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

        model = Sequential()
        model.add(Merge([image_model, lang_model], mode='concat'))
        model.add(LSTM(1000, return_sequences=False))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))

        print("Model created!")

        if(ret_model == True):
            return model

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_word(self, index):
        return self.index_word[index]

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


def train_model(weight=None, batch_size=32, epochs=10):
    cg = CaptionGenerator()
    model = cg.create_model()
    if weight != None:
        model.load_weights(weight)

    df = pd.read_csv('PR_test_dataset.txt', delimiter='\t')
    train_num = cg.total_samples
    test_num = df.shape[0]
    steps_per_epoch = train_num // batch_size
    test_steps = test_num // batch_size
    print("进入训练状态, 训练样本数/测试样本数=%d/%d" % (train_num, test_num))

    file_name = 'weights-improvement-{epoch:02d}.hdf5'
    #file_name  = 'weights.{epoch:02d-{val_loss:.2f}}.hdf5'
    checkpoint = ModelCheckpoint(
        file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', verbose=1, factor=0.5, patience=1, min_lr=0.000001)

    callbacks_list = [reduce_lr, checkpoint]
    model.fit_generator(cg.train_data_generator(batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        validation_data=cg.test_data_generator(
                            batch_size=batch_size),
                        validation_steps=test_steps,
                        callbacks=callbacks_list,
                        epochs=epochs)
    try:
        model.save('Models/WholeModel.h5', overwrite=True)
        model.save_weights('Models/Weights.h5', overwrite=True)
    except:
        print("Error in saving model.\n")
    print("Training complete...\n")

train_model(epochs=50)
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence
from keras.callbacks import ModelCheckpoint
import pickle
import sys

EMBEDDING_DIM = 128


class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.f_train_dataset_name = 'PR_train_dataset.txt'
        self.f_test_dataset_name = 'PR_test_dataset.txt'
        self.encoded_images = pickle.load(open("encoded_images.p", "rb"))
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv(self.f_train_dataset_name, delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = next(iter)
            caps.append(x[1][1])

        self.total_samples = 0
        for text in caps:
            self.total_samples += len(text.split()) - 1
        print("Total samples : " + str(self.total_samples))

        # 在数过了训练样本的个数后，应该补上测试样本，来扩充词表
        # 不过这时候就不影响total_samples了，total_samples仍然是训练样本个数。
        test_df = pd.read_csv(self.f_test_dataset_name, delimiter='\t')
        test_nb_samples = test_df.shape[0]
        test_iter = test_df.iterrows()
        for i in range(test_nb_samples):
            x = next(test_iter)
            caps.append(x[1][1])

        # words集合取unique，构建单词表
        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}

        # 对词语编号应该从1开始，而不是0
        # 这是因为0要留给caption向量作padding zero用
        # 所以下面word对应的位置是i+1，而不是i
        for i, word in enumerate(unique):
            self.word_index[word] = i+1
            self.index_word[i+1] = word

        # 更新caption值
        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print("Vocabulary size: " + str(self.vocab_size))
        print("Maximum caption length: " + str(self.max_cap_len))
        print("Variables initialization done!")

    def train_data_generator(self, batch_size=32):
        partial_caps = []
        next_words = []
        images = []
        print("Generating data...")
        gen_count = 0
        df = pd.read_csv(self.f_train_dataset_name, delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = next(iter)
            caps.append(x[1][1])
            imgs.append(x[1][0])

        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter += 1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split()) - 1):
                    total_count += 1

                    #partial = [self.word_index[txt] for txt in text.split()[:i+1]]

                    partial = []
                    for txt in text.split()[:i + 1]:
                        try:
                            partial.append(self.word_index[txt])
                        except KeyError:  # trainset里面找不到 txt 这个字
                            # 那就把这个字打印到屏幕上去
                            print("Info (train-txt): trainset has no ", txt)
                            # 然后再用一个trainset里面有的'某'字代替
                            partial.append(self.word_index['某'])

                    partial_caps.append(partial)
                    nextItem = np.zeros(self.vocab_size)
                    try:
                        # trainset里面找不到 text.split()[i+1] 这个字
                        nextItem[self.word_index[text.split()[i + 1]]] = 1
                    except KeyError:
                        # 那就把这个字打印到屏幕上去
                        print("Info (train-split): trainset has no ",
                              text.split()[i + 1])
                        # 然后再用一个trainset里面有的'某'字代替
                        nextItem[self.word_index['某']] = 1
                    next_words.append(nextItem)
                    images.append(current_image)

                    if total_count >= batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(
                            partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count += 1
                        if gen_count % 50 == 0:
                            print("yielding count: " + str(gen_count))
                            sys.stdout.flush()

                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []

    def test_data_generator(self, batch_size=32):
        partial_caps = []
        next_words = []
        images = []
        print("Generating data...")
        gen_count = 0
        df = pd.read_csv(self.f_test_dataset_name, delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = next(iter)
            caps.append(x[1][1])
            imgs.append(x[1][0])

        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter += 1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split()) - 1):
                    total_count += 1
                    #partial = [self.word_index[txt] for txt in text.split()[:i+1]]

                    partial = []
                    for txt in text.split()[:i + 1]:
                        try:
                            partial.append(self.word_index[txt])
                        except KeyError:  # trainset里面找不到 txt 这个字
                            # 那就把这个字打印到屏幕上去
                            print("Info (test-txt): trainset has no ", txt)
                            # 然后再用一个trainset里面有的'某'字代替
                            partial.append(self.word_index['某'])

                    partial_caps.append(partial)
                    nextItem = np.zeros(self.vocab_size)

                    try:
                        # trainset里面找不到 text.split()[i+1] 这个字
                        nextItem[self.word_index[text.split()[i + 1]]] = 1
                    except KeyError:
                        # 那就把这个字打印到屏幕上去
                        print("Info (test-split): trainset has no ",
                              text.split()[i + 1])
                        # 然后再用一个trainset里面有的'某'字代替
                        nextItem[self.word_index['某']] = 1

                    next_words.append(nextItem)
                    images.append(current_image)

                    if total_count >= batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(
                            partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count += 1
                        if gen_count % 50 == 0:
                            print("yielding count: " + str(gen_count))
                            sys.stdout.flush()

                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []

    def load_image(self, path):
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.asarray(x)

    def create_model(self, ret_model=False):
        #base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
        # base_model.trainable=False
        image_model = Sequential()
        # image_model.add(base_model)
        # image_model.add(Flatten())
        image_model.add(Dense(EMBEDDING_DIM, input_dim=4096, activation='relu'))
        image_model.add(RepeatVector(self.max_cap_len))
        lang_model = Sequential()
        lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len, mask_zero=True))
        lang_model.add(LSTM(256, return_sequences=True))
        lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

        model = Sequential()
        model.add(Merge([image_model, lang_model], mode='concat'))
        model.add(LSTM(1000, return_sequences=False))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))

        print("Model created!")

        if(ret_model == True):
            return model

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_word(self, index):
        return self.index_word[index]

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


def train_model(weight=None, batch_size=32, epochs=10):
    cg = CaptionGenerator()
    model = cg.create_model()
    if weight != None:
        model.load_weights(weight)

    df = pd.read_csv('PR_test_dataset.txt', delimiter='\t')
    train_num = cg.total_samples
    test_num = df.shape[0]
    steps_per_epoch = train_num // batch_size
    test_steps = test_num // batch_size
    print("进入训练状态, 训练样本数/测试样本数=%d/%d" % (train_num, test_num))

    file_name = 'weights-improvement-{epoch:02d}.hdf5'
    #file_name  = 'weights.{epoch:02d-{val_loss:.2f}}.hdf5'
    checkpoint = ModelCheckpoint(
        file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', verbose=1, factor=0.5, patience=1, min_lr=0.000001)

    callbacks_list = [reduce_lr, checkpoint]
    model.fit_generator(cg.train_data_generator(batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        validation_data=cg.test_data_generator(
                            batch_size=batch_size),
                        validation_steps=test_steps,
                        callbacks=callbacks_list,
                        epochs=epochs)
    try:
        model.save('Models/WholeModel.h5', overwrite=True)
        model.save_weights('Models/Weights.h5', overwrite=True)
    except:
        print("Error in saving model.\n")
    print("Training complete...\n")

train_model(weight=None, batch_size=2048, epochs=10)
