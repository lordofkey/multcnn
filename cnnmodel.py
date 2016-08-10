class models(object):
    def __init__(self):
        self.name = ''
        self.type = ''
        self.labels = []
        self.model = []
        self.tf_param = []





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
modelnum = 1
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