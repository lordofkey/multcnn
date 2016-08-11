#coding:utf-8
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

addr = ''

def receivedata():
    while True:
        tmp = Qcon.get()
        conn = tmp[0]
        process_num = tmp[1]
        param = conn.recv(PARAM_LEN)
        ############################################
        try:
            conn.sendall('s')
        except:
            continue
        width = struct.unpack('L', conn.recv(8))[0]
        height = struct.unpack('L', conn.recv(8))[0]
        file_size = width * height
        recv_size = 0
        data = ''
        try:
            while recv_size < file_size:
                if file_size - recv_size > 10240:
                    temp_recv = conn.recv(10240)
                    data += temp_recv
                else:
                    temp_recv = conn.recv(file_size - recv_size)
                    data += temp_recv
                recv_size = len(data)
        except:
            continue
        img = np.fromstring(data,dtype=np.uint8)
        cv2.imwrite('test1.jpg',img)
        img = img.reshape(height,width)
        img_in = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT))
        Qs.put((img_in, process_num))
        ##################################################################################
        m_rlt = ''
        try:
            conn.sendall(m_rlt)
        except:
            continue
        conn.close()

def imgpro():
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(SPATH)
    while True:
        tmp = Qs.get()
        img_in = tmp[0]
        process_num = tmp[1]
        sock.sendall(img_in.data.__str__())


def updateshow():
    while True:
        connum = Qcon.qsize()
        impronum = Qs.qsize()
        time.sleep(0.2)
        os.system('clear')
        print '################################################################################'
        print '#      接收列队负载：', connum
        print '#      处理列队负载：', impronum
        print '#        客户机地址：', addr



HOST = '0.0.0.0'
PORT = 8145
PARAM_LEN = 128

SAVE_IMG = 1
picFolder = ''

m_date = str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))



sthread = threading.Thread(target=imgpro)
sthread.setDaemon(True)
sthread.start()

sthread = threading.Thread(target=updateshow)
sthread.setDaemon(True)
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






