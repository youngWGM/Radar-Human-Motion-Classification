#!/usr/bin/env python
# _*_coding:utf-8_*_
import tensorflow as tf
import random
import os
import cv2


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def transform_label(imgType):
    label_dict = {
        'class1': 0,
        'class5': 1,
        'class6': 2,
        'class7': 3,
        'class8': 4,
        'class9': 5,
        'class10': 6,
        'class11': 7,
    }
    return label_dict[imgType]


def creat(HOME_PATH):
    totalList = list()
    filenameTrain = 'TFRecord128_random/train.tfrecords'
    filenameTest = 'TFRecord128_random/test.tfrecords'
    writerTrain = tf.python_io.TFRecordWriter(filenameTrain)
    writerTest = tf.python_io.TFRecordWriter(filenameTest)
    folders = os.listdir(HOME_PATH)
    for subFoldersName in folders:
        path = os.path.join(HOME_PATH, subFoldersName)  # 文件夹路径
        subFoldersNameList = os.listdir(path)
        for imageName in subFoldersNameList:
            imagePath = os.path.join(path, imageName)
            totalList.append(imagePath)
    dictlist = random.sample(range(0, len(totalList)), len(totalList))
    print(totalList[0].split('\\')[1].split('-')[0])

    i = 0
    for path in totalList:
        images = cv2.imread(totalList[dictlist[i]])
        res = cv2.resize(images, (64, 64), interpolation=cv2.INTER_CUBIC)
        image_raw_data = res.tostring()
        label = transform_label(totalList[dictlist[i]].split('\\')[1].split('-')[0])
        print(label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw_data)
        }))
        if i <= len(totalList) * 3 / 4:
            writerTrain.write(example.SerializeToString())
        else:
            writerTest.write(example.SerializeToString())
        i += 1
    writerTrain.close()
    writerTest.close()


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    # transform('Fnt/')
    creat('Fnt/')
