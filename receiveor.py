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






HOST = '0.0.0.0'
PORT = 8145
PARAM_LEN = 128

SAVE_IMG = 1
picFolder = ''




class models:
    def __init__(self):
        self.name = ''
        self.type = ''
        self.labels = []
        self.model = []
        self.tf_param = []   #pred, x, keep_prob
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
def impro():
    while True:
        tmp = Qs.get()
        img_in = tmp[0]
        used_model = tmp[1]
        starttime = datetime.datetime.now()
        predictions = used_model.model[0].predict([img_in])
        endtime = datetime.datetime.now()
        print endtime - starttime

sthread = threading.Thread(target=impro)
sthread.setDaemon(True)
sthread.start()

process_num = 0

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(1)
model_index = 0
while 1:
    conn, addr = s.accept()
    print'Connected by', addr
    param = conn.recv(PARAM_LEN)
    for i in range(modelnum):
        if param[:4] == classifier[i].name[:4]:
            model_index = i
    ############################################
    conn.sendall('s')
    width = struct.unpack('L', conn.recv(8))[0]
    height = struct.unpack('L', conn.recv(8))[0]
    file_size = width * height
    recv_size = 0
    im = []
    while recv_size < file_size:
        if file_size - recv_size > 1024:
            temp_recv = conn.recv(1024)
            data = list(struct.unpack(str(len(temp_recv)) + 'B', temp_recv))
            im.extend(data)
        else:
            temp_recv = conn.recv(file_size - recv_size)
            data = list(struct.unpack(str(len(temp_recv)) + 'B', temp_recv))
            im.extend(data)
        recv_size += len(data)

    img = np.array(im, np.uint8)
    img = img.reshape(height, width, 1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_in = cv2.resize(img, (227, 227))
    img_in = img_in.astype(np.float32)
    img_in /= 255
    # img_in = [skimage.img_as_float(img).astype(np.float32)]
    m_rlt = ''
    used_model = classifier[model_index]
    if used_model.type == 'caffe':
        Qs.put((img_in, used_model))
    if used_model.type == 'tensorflow':
        predictions = used_model.model[0].run(used_model.tf_param[0],
                                              feed_dict={used_model.tf_param[1]: [img_in], used_model.tf_param[2]: 1.})
        m_rlt = used_model.labels[np.argmax(predictions)]
        print predictions, m_rlt

    if(0 == process_num % 2000):
        picFolder = str(process_num)

    process_num += 1
    print process_num

  # cv2.imshow("1", img)
  # if SAVE_IMG:
  #     commands.getstatusoutput('mkdir -p pic/' + classifier[model_index].name + '/' + m_date + '/' + m_rlt + '/' + picFolder)
  ##     cv2.imwrite('pic/' + classifier[model_index].name + '/' + m_date + '/' + m_rlt + '/' + picFolder + '/' + str(process_num) + '.jpg', img)
  # cv2.waitKey(1)

    ############################################
    conn.sendall(m_rlt)
    conn.close()
