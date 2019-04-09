#_*_coding:utf-8
from PIL import Image
import os

fin = 'G:/资料/文献资料/数据/可用RadarData/11'
fout = 'G:/资料/文献资料/数据/可用RadarData/11'
for file in os.listdir(fin):
    file_fullname = fin + '/' +file
    img = Image.open(file_fullname)
    xsize,ysize = img.size
    a = [0, 0, xsize, ysize/2]
    box = (a)
    #out = img.resize((128, 128), Image.ANTIALIAS)
    roi = img.crop(box)
    #if fout not in os.listdir('E:/资料/文献/数据/train-RadarData/1'):
       # os.mkdir(fout)
    out_path = fout + '/' + file
    roi.save(out_path)