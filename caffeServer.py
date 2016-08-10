import socket
import cv2
import numpy as np
import xml.dom.minidom
import caffe
import datetime


class models(object):
    def __init__(self, num = 1):
        self.name = ''
        self.type = ''
        self.labels = []
        self.model = []
        self.tf_param = []  #pred, x, keep_prob
        self.initmodel(num)
        self.initlabel()
    def initmodel(self, num):
        dom = xml.dom.minidom.parse('config.xml')
        root = dom.documentElement
        m_mdlnum = root.getElementsByTagName('modelnum')
        itemlist = root.getElementsByTagName('model')
        model_content = itemlist[num]
        self.name = str(model_content.getAttribute("name"))
        self.type self= str(model_content.getAttribute("type"))
        model_path = str(model_content.getAttribute("path"))
        if self.type == 'caffe':
            caffe.set_mode_gpu()
            proto_data = open(model_path + 'mean.binaryproto', 'rb').read()
            temp_a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
            mean = caffe.io.blobproto_to_array(temp_a)[0]
            # mean = np.load(model_path + 'mean.npy')
            self.model.append(caffe.Classifier(model_path + 'deploy.prototxt', model_path + 'model.caffemodel',
                                            mean=mean, channel_swap=(2, 1, 0), raw_scale=255, image_dims=(227, 227)))
            print 'caffe done!'
    def initlabel(self):
        f = open(model_path + 'labels.txt', 'r')
        while True:
            line = f.readline()
            line = line.strip('\n')
            # if SAVE_IMG:
                # commands.getstatusoutput('mkdir -p pic/' + model.name + '/' + m_date + '/' + line)
            if line:
                self.labels.append(line)
            else:
                break


m_date = str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))


caffe = models()
