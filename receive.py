import socket
import cv2
import numpy as np
import struct
import xml.dom.minidom
import caffe
import tensorflow as tf
import commands
import datetime
import Queue
import threading


Qs = Queue.Queue(100)
Qcon = Queue.Queue(30)
IMG_WIDTH = 227
IMG_HEIGHT = 227
IMG_LEN = IMG_WIDTH*IMG_HEIGHT
SPATH = '/tmp/caffeServer.d'


def receivedata():
    while True:
        tmp = Qcon.get()
        conn = tmp[0]
        process_num = tmp[1]
        param = conn.recv(PARAM_LEN)
        for i in range(modelnum):
            if param[:4] == classifier[i].name[:4]:
                model_index = i
        ############################################
        try:
            conn.sendall('s')
        except:
            print "failed to receive img!"
            continue
        width = struct.unpack('L', conn.recv(8))[0]
        height = struct.unpack('L', conn.recv(8))[0]
        file_size = width * height
        recv_size = 0
        im = []
        try:
            while recv_size < file_size:
                if file_size - recv_size > 10240:
                    temp_recv = conn.recv(10240)
                    data = list(struct.unpack(str(len(temp_recv)) + 'B', temp_recv))
                    im.extend(data)
                else:
                    temp_recv = conn.recv(file_size - recv_size)
                    data = list(struct.unpack(str(len(temp_recv)) + 'B', temp_recv))
                    im.extend(data)
                recv_size += len(data)
        except:
            continue
        Qs.put((im, process_num, width, height))
        ##################################################################################
        m_rlt = ''
        try:
            conn.sendall(m_rlt)
        except:
            print "failed to sentback result!"
            continue
        conn.close()






HOST = '0.0.0.0'
PORT = 8145
PARAM_LEN = 128

SAVE_IMG = 1
picFolder = ''

class models(object):
    def __init__(self):
        self.name = ''
        self.type = ''
        self.labels = []
        self.model = []
        self.tf_param = []  #pred, x, keep_prob
####PreProcess####
m_date = str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
sess = tf.Session()

classifier = []
labels = []
dom = xml.dom.minidom.parse('config.xml')
root = dom.documentElement
m_mdlnum = root.getElementsByTagName('modelnum')
m_mdlnum_temp = m_mdlnum[0].firstChild.data
modelnum = int(m_mdlnum_temp)
itemlist = root.getElementsByTagName('model')
#modelnum = 1
for ii in range(modelnum):
    model_content = itemlist[ii]
    model = models()
    model.name = str(model_content.getAttribute("name"))
    model.type = str(model_content.getAttribute("type"))
    model_path = str(model_content.getAttribute("path"))
    if model.type == 'caffe':
        caffe.set_mode_gpu()
        proto_data = open(model_path + 'mean.binaryproto', 'rb').read()
        temp_a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        mean = caffe.io.blobproto_to_array(temp_a)[0]
        # mean = np.load(model_path + 'mean.npy')
        model.model.append(caffe.Classifier(model_path + 'deploy.prototxt', model_path + 'model.caffemodel',
                                            mean=mean, channel_swap=(2, 1, 0), raw_scale=255, image_dims=(227, 227)))

        print 'caffe done!'
    if model.type == 'tensorflow':
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

        pred = tf.get_collection("pred")[0]
        x = tf.get_collection("x")[0]
        keep_prob = tf.get_collection("keep_prob")[0]
        model.tf_param.append(pred)
        model.tf_param.append(x)
        model.tf_param.append(keep_prob)

        saver.restore(sess, ckpt.model_checkpoint_path)
        model.model.append(sess)
        print 'tf done!'

    f = open(model_path + 'labels.txt', 'r')
    while True:
        line = f.readline()
        line = line.strip('\n')
        # if SAVE_IMG:
            # commands.getstatusoutput('mkdir -p pic/' + model.name + '/' + m_date + '/' + line)
        if line:
            model.labels.append(line)
        else:
            break
    f.close()
    classifier.append(model)

def imgpro(classifier):
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(SPATH)

    while True:
        tmp = Qs.get()
        im = tmp[0]
        process_num = tmp[1]
        width = tmp[2]
        height = tmp[3]
        img = np.array(im, np.uint8)
        img = img.reshape(height, width, 1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_in = cv2.resize(img, (227, 227))
        buff = struct.pack(str(IMG_LEN) + 'B', img_in)
        sock.sendall(buff)
        m_rlt = ''
        if(0 == process_num % 2000):
            picFolder = str(process_num)
        print process_num
#        cv2.imshow("1", img)
        if SAVE_IMG:
            commands.getstatusoutput('mkdir -p pic/' + classifier[model_index].name + '/' + m_date + '/' + m_rlt + '/' + picFolder)
            cv2.imwrite('pic/' + classifier[model_index].name + '/' + m_date + '/' + m_rlt + '/' + picFolder + '/' + str(process_num) + '.jpg', img)
#        cv2.waitKey(1)

for i in range(1):
    sthread = threading.Thread(target=imgpro, args=(classifier,))
    sthread.setDaemon(False)
    sthread.start()

for i in range(10):
    sthread = threading.Thread(target=receivedata)
    sthread.setDaemon(True)
    sthread.start()


s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(1)
model_index = 0
process_num = 0
while True:
    conn, addr = s.accept()
    print 'Connected by', addr
    Qcon.put((conn, process_num))
    process_num += 1






