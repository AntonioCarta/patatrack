# flake8: noqa: E402, F401
import socket
if socket.gethostname() == 'cmg-gpu1080':
    print('locking only one GPU.')
    import setGPU

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


target_lab = "pdgId"

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

headLab = ["run", "evt", "detSeqIn", "detSeqOut", "inX", "inY", "inZ", "outX", "outY", "outZ",
           "inPhi", "inR", "outPhi", "outR",
           "detCounterIn", "detCounterOut", "isBarrelIn", "isBarrelOut",
           "layerIn", "ladderIn", "moduleIn", "sideIn", "diskIn", "panelIn", "bladeIn",
           "layerOut", "ladderOut", "moduleOut", "sideOut", "diskOut", "panelOut", "bladeOut",
           "isBigIn", "isEdgIn", "isBadIn", "isBigOut", "isEdgOut", "isBadOut",
           "isFlippedIn", "isFlippedOut",
           "iCSize", "pixInX", "pixInY", "inClusterADC", "iZeroADC", "iCSize2", "iCSizeX", "iCSizeY", "iCSizeY2",
           "iOverFlowX", "iOverFlowY",
           "oCSize", "pixOutX", "pixOutY", "outClusterADC", "oZeroADC", "oCSize2", "oCSizeX", "oCSizeY", "oCSizeY2",
           "oOverFlowX", "oOverFlowY",
           "diffADC"
           ]

inhitlabs = ["inPix" + str(el) for el in range(1, 226)]
outhitlabs = ["outPix" + str(el) for el in range(1, 226)]


targetlabes = ["idTrack", "px", "py", "pz", "pt", "mT", "eT", "mSqr", "rapidity", "etaTrack", "phi",
               "pdgId", "charge", "noTrackerHits", "noTrackerLayers", "dZ", "dXY", "Xvertex",
               "Yvertex", "Zvertex", "bunCross", "isCosmic", "chargeMatch", "sigMatch"
               ]

XYZ = ["X", "Y", "Z"]
inXYZ = ["inX", "inY", "inZ"]
outXYZ = ["outX", "outY", "outZ"]

featurelabs = [
    "detSeqIn", "detSeqOut", "inX", "inY", "inZ", "outX", "outY", "outZ",
    "inPhi", "inR", "outPhi", "outR",
    "detCounterIn", "detCounterOut", "isBarrelIn", "isBarrelOut",
    "layerIn", "ladderIn", "moduleIn", "sideIn", "diskIn", "panelIn", "bladeIn",
    "layerOut", "ladderOut", "moduleOut", "sideOut", "diskOut", "panelOut", "bladeOut",
    "isBigIn", "isEdgIn", "isBadIn", "isBigOut", "isEdgOut", "isBadOut",
    "isFlippedIn", "isFlippedOut",
    "iCSize", "pixInX", "pixInY", "inClusterADC", "iZeroADC", "iCSize", "iCSizeX", "iCSizeY",
    "iOverFlowX", "iOverFlowY",
    "oCSize", "pixOutX", "pixOutY", "outClusterADC", "oZeroADC", "oCSize", "oCSizeX", "oCSizeY",
    "oOverFlowX", "oOverFlowY",
    "diffADC"
]

dataLab = headLab + inhitlabs + outhitlabs + ["dummyFlag"] + targetlabes
doubLab = headLab + inhitlabs + outhitlabs + ["dummyFlag"]

infoLab = XYZ + targetlabes


class Dataset:
    """ Load the dataset from txt files. """

    def __init__(self, fnames):
        self.data = pd.DataFrame(data=[], columns=dataLab)
        for f in fnames:
            print("Loading: " + f)
            df = pd.read_hdf(f, mode='r')
            df.columns = dataLab  # change wrong columns names
            self.data = self.data.append(df)

    def from_dataframe(data):
        """ Constructor method to initialize the classe from a DataFrame """
        d = Dataset([])
        d.data = data
        return d

    def theta_correction(self, hits_in, hits_out):
        # theta correction
        cosThetaIns = np.cos(np.arctan(np.multiply(
            self.data["inY"], 1.0 / self.data["inZ"])))
        cosThetaOuts = np.cos(np.arctan(np.multiply(
            self.data["outY"], 1.0 / self.data["outZ"])))
        sinThetaIns = np.sin(np.arctan(np.multiply(
            self.data["inY"], 1.0 / self.data["inZ"])))
        sinThetaOuts = np.sin(np.arctan(np.multiply(
            self.data["outY"], 1.0 / self.data["outZ"])))

        inThetaModC = np.multiply(hits_in, cosThetaIns[:, np.newaxis])
        outThetaModC = np.multiply(hits_out, cosThetaOuts[:, np.newaxis])

        inThetaModS = np.multiply(hits_in, sinThetaIns[:, np.newaxis])
        outThetaModS = np.multiply(hits_out, sinThetaOuts[:, np.newaxis])
        return inThetaModC, outThetaModC, inThetaModS, outThetaModS

    def separate_flipped_hits(self, hit_shapes, flipped):
        flipped = flipped.astype('bool')
        flipped_hits = np.zeros(hit_shapes.shape)
        not_flipped_hits = np.zeros(hit_shapes.shape)
        flipped_hits[flipped, :] = hit_shapes[flipped, :]
        not_flipped_hits[~flipped, :] = hit_shapes[~flipped, :]
        return flipped_hits, not_flipped_hits

    def get_hit_shapes(self, normalize=True, angular_correction=True, flipped_channels=True):
        """ Return hit shape features
        Args:
        -----
            normalize : (bool)
                normalize the data matrix with zero mean and unitary variance.
        """
        a_in = self.data[inhitlabs].as_matrix()
        a_out = self.data[outhitlabs].as_matrix()

        # Normalize data
        if normalize:
            # mean, std precomputed for data NOPU
            mean, std = (668.25684, 3919.5576)
            a_in = a_in / std
            a_out = a_out / std

        if flipped_channels:
            flip_in, not_flip_in = self.separate_flipped_hits(
                a_in, self.data.isFlippedIn)
            flip_out, not_flip_out = self.separate_flipped_hits(
                a_out, self.data.isFlippedOut)
            l = [flip_in, not_flip_in, flip_out, not_flip_out]
        else:
            l = [a_in, a_out]
        if angular_correction:
            thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(
                a_in, a_out)
            l = l + [thetac_in, thetac_out, thetas_in, thetas_out]

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, 15, 15))
        # TODO: not optimal for CPU execution
        return np.transpose(data, (1, 2, 3, 0))

    def filter(self, feature_name, value):
        """ filter data keeping only those samples where s[feature_name] = value """
        self.data = self.data[self.data[feature_name] == value]
        return self  # to allow method chaining

    def get_info_features(self):
        """ Returns info features as numpy array. """
        return self.data[featurelabs].as_matrix()

    def get_layer_map_data(self, crop=False):
        a_in = self.data[inhitlabs].as_matrix().astype(np.float16)
        a_out = self.data[outhitlabs].as_matrix().astype(np.float16)

        # mean, std precomputed for data NOPU
        mean, std = (668.25684, 3919.5576)
        a_in = a_in / std
        a_out = a_out / std

        l = []
        thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(
            a_in, a_out)
        l = l + [thetac_in, thetac_out, thetas_in, thetas_out]

        for hits, ids in [(a_in, self.data.detSeqIn), (a_out, self.data.detSeqOut)]:
            for id_layer in layer_ids:
                layer_hits = np.zeros(hits.shape)
                bool_mask = ids == id_layer
                layer_hits[bool_mask, :] = hits[bool_mask, :]
                l.append(layer_hits)

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, 15, 15))
        X_hit = np.transpose(data, (1, 2, 3, 0))

        if crop:
            X_hit = X_hit[:, 4:11, 4:11, :]
        X_info = self.get_info_features()
        y = to_categorical(self.get_labels())
        return X_hit, X_info, y

    def get_labels(self):
        return self.data[target_lab].as_matrix() != -1.0

    def get_data(self, normalize=True, angular_correction=True, flipped_channels=True, crop=False):
        X_hit = self.get_hit_shapes(
            normalize, angular_correction, flipped_channels)
        if crop:
            X_hit = X_hit[:, 4:11, 4:11, :]
        X_info = self.get_info_features()
        y = to_categorical(self.get_labels(), num_classes=2)
        return X_hit, X_info, y

    def save(self, fname):
        # np.save(fname, self.data.as_matrix())
        self.data.to_hdf(fname, 'data', mode='w')

    # TODO: pick doublets from same event.
    def balance_data(self, max_ratio=0.5, verbose=True):
        """ Balance the data. """
        data_neg = self.data[self.data[target_lab] == -1.0]
        data_pos = self.data[self.data[target_lab] != -1.0]

        n_pos = data_pos.shape[0]
        n_neg = data_neg.shape[0]

        if verbose:
            print("Number of negatives: " + str(n_neg))
            print("Number of positive: " + str(n_pos))
            print("ratio: " + str(n_neg / n_pos))

        if n_pos > n_neg:
            return self

        data_neg = data_neg.sample(n_pos)
        balanced_data = pd.concat([data_neg, data_pos])
        balanced_data = balanced_data.sample(frac=1)  # Shuffle the dataset
        self.data = balanced_data
        return self  # allow method chaining


if __name__ == '__main__':
    d = Dataset('data/debug.npy')
    batch_size = d.data.as_matrix().shape[0]

    x = d.get_data()
    assert x[0].shape == (batch_size, 15, 15, 8)

    x = d.get_data(normalize=False, angular_correction=False,
                   flipped_channels=False)[0]
    assert x.shape == (batch_size, 15, 15, 2)
    np.testing.assert_allclose(
        x[:, :, :, 0], d.data[inhitlabs].as_matrix().reshape((-1, 15, 15)))

    print("All test successfully completed.")
