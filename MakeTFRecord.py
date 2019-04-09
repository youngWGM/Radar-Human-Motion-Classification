# _*_coding:utf-8_*_
import tensorflow as tf
import numpy as np
import os
import cv2


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def transform_label(folderName):
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
    return label_dict[folderName]


def transform(HOME_PATH):
    filenameTrain = 'TFRecord128_5/train.tfrecords'
    filenameTest = 'TFRecord128_5/test.tfrecords'
    writerTrain = tf.python_io.TFRecordWriter(filenameTrain)
    writerTest = tf.python_io.TFRecordWriter(filenameTest)
    folders = os.listdir(HOME_PATH)
    for subFoldersName in folders:
        label = transform_label(subFoldersName)
        path = os.path.join(HOME_PATH, subFoldersName)  # 文件夹路径
        subFoldersNameList = os.listdir(path)
        i = 0
        for imageName in subFoldersNameList:
            imagePath = os.path.join(path, imageName)
            images = cv2.imread(imagePath)
            res = cv2.resize(images, (128, 128), interpolation=cv2.INTER_CUBIC)
            image_raw_data = res.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_raw_data)
            }))
            #if i >= len(subFoldersNameList) * 1 / 2:
            writerTrain.write(example.SerializeToString())
            #else:
                #writerTest.write(example.SerializeToString())
            #i += 1
    writerTrain.close()
    writerTest.close()


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    transform('F:/pycharm my project/Radar2/RadarData/')
