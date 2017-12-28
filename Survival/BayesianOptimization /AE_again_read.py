from keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split 
from keras import backend as K
from keras.layers import Input, Dense 
from keras.models import Model
import os  
###### input parameters 
#cancer_type = "LUAD"
#feature_type = "RPM_miRNA.hg19.mirbase20"

def AE_model_save(cancer_type, feature_type,dim1,dim2, x_trn, x_tst):

    """
    Input data
    if not (os.path.exists('../data/%s.%s.pkl' % (cancer_type, feature_type))):
        return None; 
    with open('../data/%s.%s.pkl' % (cancer_type, feature_type), 'rb') as handle:
            labels = pickle.load(handle)#,encoding = 'latin1')
            x = pickle.load(handle)#,encoding = 'latin1')
            y = pickle.load(handle)#,encoding = 'latin1')

    x_trn, x_tst, c_trn, c_tst, s_trn, s_tst, l_trn, l_tst = \
            train_test_split(x, y[:, 0], y[:, 1], labels, test_size=80, random_state=7)
    x_trn, x_dev, c_trn, c_dev, s_trn, s_dev, l_trn, l_dev= \
                train_test_split(x_trn, c_trn, s_trn, l_trn, test_size=20, random_state=7)
    """
    AE_file_path = './model/AE_%s-%s-%d-%d.h5py' % (cancer_type, feature_type, dim1, dim2)
    E_file_path = './model/E_%s-%s-%d-%d.h5py' % (cancer_type, feature_type, dim1, dim2)
    if os.path.exists(AE_file_path) and os.path.exists(E_file_path):
        # just read model and predict 
        encoder = load_model(E_file_path)
        return encoder.predict(x_trn), encoder.predict(x_tst)
    feature_num =x_trn.shape[1] 
    hidden_dim_1 = dim1
    hidden_dim_2 = dim2
    input_shape = (feature_num, )
    print("AE input shape "+ str(input_shape))
    input_layer = Input(shape=input_shape)
    #input_layer = Input(shape=x_trn.shape[1:])

    encoded = Dense(hidden_dim_1, activation='relu')(input_layer)
    encoded = Dense(hidden_dim_2, activation ='relu')(encoded)

    decoded = Dense(hidden_dim_1, activation='relu')(encoded)
    decoded = Dense(feature_num, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_trn, x_trn, epochs=100, batch_size=32, shuffle=True, validation_split=0.2)
    reduced_X_trn = encoder.predict(x_trn).astype(np.float64)
    reduced_X_tst = encoder.predict(x_trn).astype(np.float64)
    
    autoencoder.save(AE_file_path,overwrite=True)
    encoder.save(E_file_path,overwrite=True)

    
    #reconstructed_X = autoencoder.predict(x_trn).astype(np.float64)
    #mse = np.mean((x_trn-reconstructed_X)**2)
    #print(mse)
    #print(x_tst.shape)
    #print(reduced_X.shape)
    return reduced_X_trn, reduced_X_tst 

"""
results = []
for ct in ['LUAD', 'LUSC']:
    for ft in ['normalized_RNA-seq.hg19', 'RPM_miRNA.hg19.mirbase20', 'betaValue_methyl.hg19.sd15']:
        for i in range(100, 1000, 200):
            results.append("%s | %s :: hidden_dim1 %d, hidden_dim2 %d mse %d"%( 
             ct,ft,i, i/2, AE_mse(ct, ft, i,int(i/2)))) 
print(results) 
"""            
        
