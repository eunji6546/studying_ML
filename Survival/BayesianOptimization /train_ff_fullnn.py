import AE_again_read
from keras.layers import Dense, Dropout, Input
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import ModelCheckpoint
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.decomposition import PCA 


AE_FLAG="AE" # Else "" 

def cindex(cen, surv, y_pred):
    N = surv.shape[0]
    Comparable = np.zeros([N,N])

    for i in range(N):
        for j in range(N):
            if cen[i] == 0 and cen[j] == 0:
                if surv[i] != surv[j]:
                    Comparable[i, j] = 1

            elif cen[i] == 1 and cen[j] == 1:
                Comparable[i, j] = 0

            else: # one sample is censored and the other is not
                if cen[i] == 1:
                    if surv[i] >= surv[j]:
                        Comparable[i, j] = 1
                    else:
                        Comparable[i, j] = 0
                else: # cen[j] == 1
                    if surv[j] >= surv[i]:
                        Comparable[i, j] = 1
                    else:
                        Comparable[i, j] = 0

    p2, p1 = np.where(Comparable==1)
    Y = y_pred

    c=0
    N_valid_sample = p1.shape[0]
    for i in range(N_valid_sample):
        if cen[p1[i]] == 0 and cen[p2[i]] == 0:
            if Y[p1[i]] == Y[p2[i]]:
                c = c + 0.5
            elif Y[p1[i]] > Y[p2[i]] and surv[p1[i]] > surv[p2[i]]:
                c = c + 1
            elif Y[p2[i]] > Y[p1[i]] and surv[p2[i]] > surv[p1[i]]:
                c = c + 1

        elif cen[p1[i]] == 1 and cen[p2[i]] == 1:
            continue # do nothing - samples cannot be ordered

        else: # one sample is censored and the other is not
            if cen[p1[i]] == 1:
                if Y[p1[i]] > Y[p2[i]] and surv[p1[i]] > surv[p2[i]]:
                    c = c + 1
                elif Y[p1[i]] == Y[p2[i]]:
                    c = c + 0.5

            else: # cen[p2[i]] == 1
                if Y[p2[i]] > Y[p1[i]] and surv[p2[i]] > surv[p1[i]]:
                    c = c + 1
                elif Y[p1[i]] == Y[p2[i]]:
                    c = c + 0.5

    c = c*1.0 / N_valid_sample
    return c

class MyCallback(ModelCheckpoint):
    def __init__(self, filepath, data, real_save=True, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, cancer_type='NULL', feature_type='Not', 
                 thr = 0, dropout_prob = 0.8, dimension=400, activate="relu",
                 AE1=800, AE2=400):
        super(MyCallback, self).__init__(filepath, monitor, verbose,
                                         save_best_only, save_weights_only,
                                         mode, period)

        self.x_trn, self.c_trn, self.s_trn, \
        self.x_dev, self.c_dev, self.s_dev, \
        self.x_tst, self.c_tst, self.s_tst = data

        self.score_trn = 0
        self.score_dev = 0
        self.score_tst = 0
        self.real_save = real_save
        self.cancer_type = cancer_type
        self.feature_type = feature_type
        self.threshold = thr
        self.dropout_prob = dropout_prob
        self.dim = dimension
        self.mode = activate
        self.ae1 = AE1
        self.ae2 = AE2

    def evaluate(self):
        pred_trn = self.model.predict(self.x_trn)
        cindex_trn = cindex(self.c_trn, self.s_trn, pred_trn)

        pred_dev = self.model.predict(self.x_dev)
        cindex_dev = cindex(self.c_dev, self.s_dev, pred_dev)

        pred_tst = self.model.predict(self.x_tst)
        cindex_tst = cindex(self.c_tst, self.s_tst, pred_tst)

        if self.score_dev < cindex_dev:
            self.score_trn = cindex_trn
            self.score_dev = cindex_dev
            self.score_tst = cindex_tst

            print("\nupdated!!")
            if self.real_save == True:
                if self.save_weights_only:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)

        print '\n[This Epoch]'
        print '\t'.join(map(str, [cindex_trn, cindex_dev, cindex_tst]))
        print '[Current Best]'
        print '\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst]))

    def on_train_end(self, logs=None):
        print '[Best:on_train_end]'
        print '\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst]))
        with open('../data'+AE_FLAG+'/fullNN_last/'+self.mode+'/result_pca'+str(self.threshold)
        +'drp'+str(self.dropout_prob)+'_dim'+str(self.dim)+'AE'+str(self.ae1)+':'+str(self.ae2) + '.txt', 'at') as f : 
            f.write("%s\t%s\t%s\t%s\t%s\n" %(self.cancer_type, self.feature_type, str(self.score_trn), str(self.score_dev), str(self.score_tst) ))

    def on_epoch_end(self, epoch, logs=None):
        self.evaluate()


def run(gpunum, cancer_type, feature_type, attempt, percentage, dropout_prob, dim, mode, AE1, AE2):
    
    dropout_prob = dropout_prob
    hidden_dims = dim
    batch_size = 32
    epochs = 100

    if (cancer_type == 'LUSC' and feature_type == 'betaValue_methyl.hg19.sd15') : 
        return 

    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum
    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())


    def ModelV1(model_input):
        z = Dropout(dropout_prob)(model_input)
        z = Dense(hidden_dims, activation=mode)(z)
        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation=mode)(z)
        model_output = Dense(1, activation=None)(z)

        model = Model(model_input, model_output)
        model.compile(loss="mse", optimizer="adam", metrics=["mse"])

        return model

    with open('../data/%s.%s.pkl' % (cancer_type, feature_type), 'rb') as handle:
        labels = pickle.load(handle)
        x = pickle.load(handle)
        y = pickle.load(handle)

    x_trn, x_tst, c_trn, c_tst, s_trn, s_tst, l_trn, l_tst = \
        train_test_split(x, y[:, 0], y[:, 1], labels, test_size=80, random_state=7)
    # Do Feature Extraction in HERE !! PCA or AutoEncoder 
    # First try : PCA with 99.999% 
    if (len(AE_FLAG) == 0) :
        # AE 
        x_trn, x_tst = AE_again_read.AE_model_save(cancer_type,feature_type,AE1,AE2,x_trn,x_tst)
    
    else :
        if ( percentage != 0):
            clf = PCA(percentage,whiten=True)     
            x_trn=clf.fit_transform(x_trn)
            x_tst=clf.transform(x_tst)

    x_trn, x_dev, c_trn, c_dev, s_trn, s_dev, l_trn, l_dev= \
            train_test_split(x_trn, c_trn, s_trn, l_trn, test_size=20, random_state=7)
    
    feature_dim = x_trn.shape[1]
    input_shape = (feature_dim, )
    model_input = Input(shape=input_shape)
    model = ModelV1(model_input)

    model.summary()

    model_filepath = '../model/%s-%s-%d.model' % (cancer_type, feature_type, attempt)
    data = tuple((x_trn, c_trn, s_trn, x_dev, c_dev, s_dev, x_tst, c_tst, s_tst))
    checkpoint = MyCallback(model_filepath, data, real_save=True, verbose=1,
                            save_best_only=True,
                            mode='auto', cancer_type = cancer_type , feature_type = feature_type,
                             thr=percentage, dropout_prob = dropout_prob, dimension = dim ,activate=mode,
                             AE1 = AE1, AE2=AE2)
    callbacks_list = [checkpoint]

    model.fit(x_trn, s_trn,
              batch_size=batch_size,
              shuffle=True,
              callbacks=callbacks_list,
              epochs=epochs,
              validation_data=(x_dev, s_dev))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default="3", choices=["0", "1"], type=str)
    parser.add_argument('-ct', default="LUAD", choices=['LUAD', 'LUSC'], type=str) # cancer_type
    parser.add_argument('-ft', default="RPM_miRNA.hg19.mirbase20",
                        choices=['normalized_RNA-seq.hg19', 'RPM_miRNA.hg19.mirbase20', 'betaValue_methyl.hg19.sd15'],
                        type=str)  # feature_type
    parser.add_argument('-t', default=0, choices=range(10), type=int)
    parser.add_argument('-prcnt', default=0.99999,type=float)
    
    parser.add_argument('-drp', default=0.8, type=float)
    parser.add_argument('-dim', default=400, type=int)
    parser.add_argument('-m', default = "relu", choices=["relu", "sigmoid", "softmax"], type=str) 
    args = parser.parse_args()

    run(args.g, args.ct, args.ft, args.t, args.prcnt, args.drp, args.dim, args.m, 800, 400 )
    
    for ct in['LUAD', 'LUSC']:
        for ft in ['normalized_RNA-seq.hg19', 'RPM_miRNA.hg19.mirbase20', 'betaValue_methyl.hg19.sd15'] :
            for dim in [100, 400, 800]:

                
                run(args.g, ct, ft, args.t, 0, 0, dim,'relu', 800,400)
                run(args.g, ct, ft, args.t, 0, 0, dim,'sigmoid', 800,400)
                run(args.g, ct, ft, args.t, 0, 0, dim,'softmax', 800,400)
                run(args.g, ct, ft, args.t, 0, 0, dim,'relu', 1000,400)
                run(args.g, ct, ft, args.t, 0, 0, dim,'sigmoid', 1000,400)
                run(args.g, ct, ft, args.t, 0, 0, dim,'softmax', 1000,400) 
                #run(args.g, ct, ft, args.t, 0, 0, dim,'relu',500,200)
                #run(args.g, ct, ft, args.t, 0, 0, dim,'relu',600,300)
                #run(args.g, ct, ft, args.t, 0, 0, dim,'relu',700,350)
                #run(args.g, ct, ft, args.t, 0, 0, dim,'relu',1000,500)
    
