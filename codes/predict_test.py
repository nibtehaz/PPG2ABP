"""
    Computes the outputs for test data
"""

from helper_functions import *
from models import UNetDS64, MultiResUNet1D
import os


def predict_test_data():
    """
        Computes the outputs for test data
        and saves them in order to avoid recomputing
    """

    length = 1024               # length of signal

    dt = pickle.load(open(os.path.join('data','test.p'),'rb'))      # loading test data
    X_test = dt['X_test']
    Y_test = dt['Y_test']   


    mdl1 = UNetDS64(length)                                             # creating approximation network
    mdl1.load_weights(os.path.join('models','ApproximateNetwork.h5'))   # loading weights
    
    Y_test_pred_approximate = mdl1.predict(X_test,verbose=1)            # predicting approximate abp waveform

    pickle.dump(Y_test_pred_approximate,open('test_output_approximate.p','wb')) # saving the approxmiate predictions


    mdl2 = MultiResUNet1D(length)                                       # creating refinement network
    mdl2.load_weights(os.path.join('models','RefinementNetwork.h5'))    # loading weights

    Y_test_pred = mdl2.predict(Y_test_pred_approximate[0],verbose=1)    # predicting abp waveform

    pickle.dump(Y_test_pred,open('test_output.p','wb'))                 # saving the predicted abp waeforms




def main():
    predict_test_data()     # predicts and stores the outputs of test data to avoid recomputing

if __name__ == '__main__':
    main()