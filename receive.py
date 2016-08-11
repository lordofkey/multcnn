import socket
import cv2
import numpy as np
import struct
import xml.dom.minidom
import caffe
import commands
import datetime
import Queue
import threading
import os
import time

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
            continue
        conn.close()

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
        img_in = cv2.resize(img, (227, 227))
        sock.sendall(img_in.data.__str__())


def updateshow():
    while True:
        time.sleep(0.1)
        os.system('cls')
        print '################################################################################'
        print '#'



HOST = '0.0.0.0'
PORT = 8145
PARAM_LEN = 128

SAVE_IMG = 1
picFolder = ''

m_date = str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))



sthread = threading.Thread(target=imgpro, args=(classifier,))
sthread.setDaemon(False)
sthread.start()

sthread = threading.Thread(target=updateshow)
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
    Qcon.put((conn, process_num))
    process_num += 1






