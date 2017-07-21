"""
Load all files in a directory (.gz), merge them in a single file
and split them in TRAIN / VAL / TEST data
"""
import gzip
import os
import numpy as np
import dataset
import pandas as pd

allfiles = ['1_100_Dataset.txt', '1_101_Dataset.txt', '1_102_Dataset.txt', '1_103_Dataset.txt', '1_104_Dataset.txt', '1_105_Dataset.txt', '1_106_Dataset.txt', '1_107_Dataset.txt', '1_108_Dataset.txt', '1_109_Dataset.txt', '1_10_Dataset.txt', '1_110_Dataset.txt', '1_111_Dataset.txt', '1_112_Dataset.txt', '1_113_Dataset.txt', '1_114_Dataset.txt', '1_115_Dataset.txt', '1_116_Dataset.txt', '1_117_Dataset.txt', '1_118_Dataset.txt', '1_119_Dataset.txt', '1_11_Dataset.txt', '1_120_Dataset.txt', '1_121_Dataset.txt', '1_122_Dataset.txt', '1_123_Dataset.txt', '1_124_Dataset.txt', '1_125_Dataset.txt', '1_126_Dataset.txt', '1_127_Dataset.txt', '1_128_Dataset.txt', '1_129_Dataset.txt', '1_12_Dataset.txt', '1_130_Dataset.txt', '1_131_Dataset.txt', '1_132_Dataset.txt', '1_133_Dataset.txt', '1_134_Dataset.txt', '1_135_Dataset.txt', '1_136_Dataset.txt', '1_137_Dataset.txt', '1_138_Dataset.txt', '1_139_Dataset.txt', '1_13_Dataset.txt', '1_140_Dataset.txt', '1_141_Dataset.txt', '1_142_Dataset.txt', '1_143_Dataset.txt', '1_144_Dataset.txt', '1_145_Dataset.txt', '1_146_Dataset.txt', '1_147_Dataset.txt', '1_148_Dataset.txt', '1_149_Dataset.txt', '1_14_Dataset.txt', '1_150_Dataset.txt', '1_151_Dataset.txt', '1_152_Dataset.txt', '1_153_Dataset.txt', '1_154_Dataset.txt', '1_155_Dataset.txt', '1_156_Dataset.txt', '1_157_Dataset.txt', '1_158_Dataset.txt', '1_159_Dataset.txt', '1_15_Dataset.txt', '1_160_Dataset.txt', '1_161_Dataset.txt', '1_162_Dataset.txt', '1_163_Dataset.txt', '1_164_Dataset.txt', '1_165_Dataset.txt', '1_166_Dataset.txt', '1_167_Dataset.txt', '1_168_Dataset.txt', '1_169_Dataset.txt', '1_16_Dataset.txt', '1_170_Dataset.txt', '1_171_Dataset.txt', '1_172_Dataset.txt', '1_173_Dataset.txt', '1_174_Dataset.txt', '1_175_Dataset.txt', '1_176_Dataset.txt', '1_177_Dataset.txt', '1_178_Dataset.txt', '1_179_Dataset.txt', '1_17_Dataset.txt', '1_180_Dataset.txt', '1_181_Dataset.txt', '1_182_Dataset.txt', '1_183_Dataset.txt', '1_184_Dataset.txt', '1_185_Dataset.txt', '1_186_Dataset.txt', '1_187_Dataset.txt', '1_188_Dataset.txt', '1_189_Dataset.txt', '1_18_Dataset.txt', '1_190_Dataset.txt', '1_191_Dataset.txt', '1_192_Dataset.txt', '1_193_Dataset.txt', '1_194_Dataset.txt', '1_195_Dataset.txt', '1_196_Dataset.txt', '1_197_Dataset.txt', '1_198_Dataset.txt', '1_199_Dataset.txt', '1_19_Dataset.txt', '1_1_Dataset.txt', '1_200_Dataset.txt', '1_201_Dataset.txt', '1_202_Dataset.txt', '1_203_Dataset.txt', '1_204_Dataset.txt', '1_205_Dataset.txt', '1_206_Dataset.txt', '1_207_Dataset.txt', '1_208_Dataset.txt', '1_209_Dataset.txt', '1_20_Dataset.txt', '1_210_Dataset.txt', '1_211_Dataset.txt', '1_212_Dataset.txt', '1_213_Dataset.txt', '1_214_Dataset.txt', '1_215_Dataset.txt', '1_216_Dataset.txt', '1_217_Dataset.txt', '1_218_Dataset.txt', '1_219_Dataset.txt', '1_21_Dataset.txt', '1_220_Dataset.txt', '1_221_Dataset.txt', '1_222_Dataset.txt', '1_223_Dataset.txt', '1_224_Dataset.txt', '1_225_Dataset.txt', '1_226_Dataset.txt', '1_227_Dataset.txt', '1_228_Dataset.txt', '1_229_Dataset.txt', '1_22_Dataset.txt', '1_230_Dataset.txt', '1_231_Dataset.txt', '1_232_Dataset.txt', '1_233_Dataset.txt', '1_234_Dataset.txt', '1_235_Dataset.txt', '1_236_Dataset.txt', '1_237_Dataset.txt', '1_238_Dataset.txt', '1_239_Dataset.txt', '1_23_Dataset.txt', '1_240_Dataset.txt', '1_241_Dataset.txt', '1_242_Dataset.txt', '1_243_Dataset.txt', '1_244_Dataset.txt', '1_245_Dataset.txt', '1_246_Dataset.txt', '1_247_Dataset.txt', '1_248_Dataset.txt', '1_249_Dataset.txt', '1_24_Dataset.txt', '1_250_Dataset.txt', '1_251_Dataset.txt', '1_252_Dataset.txt', '1_253_Dataset.txt', '1_254_Dataset.txt', '1_255_Dataset.txt', '1_256_Dataset.txt', '1_257_Dataset.txt', '1_258_Dataset.txt', '1_259_Dataset.txt', '1_25_Dataset.txt', '1_260_Dataset.txt', '1_261_Dataset.txt', '1_262_Dataset.txt', '1_263_Dataset.txt', '1_264_Dataset.txt', '1_265_Dataset.txt', '1_266_Dataset.txt', '1_267_Dataset.txt', '1_268_Dataset.txt', '1_269_Dataset.txt', '1_26_Dataset.txt', '1_270_Dataset.txt', '1_271_Dataset.txt', '1_272_Dataset.txt', '1_273_Dataset.txt', '1_274_Dataset.txt', '1_275_Dataset.txt', '1_276_Dataset.txt', '1_277_Dataset.txt', '1_278_Dataset.txt', '1_279_Dataset.txt', '1_27_Dataset.txt', '1_280_Dataset.txt', '1_281_Dataset.txt', '1_282_Dataset.txt', '1_283_Dataset.txt', '1_284_Dataset.txt', '1_285_Dataset.txt', '1_286_Dataset.txt', '1_287_Dataset.txt', '1_288_Dataset.txt', '1_289_Dataset.txt', '1_28_Dataset.txt', '1_290_Dataset.txt', '1_291_Dataset.txt', '1_292_Dataset.txt', '1_293_Dataset.txt', '1_294_Dataset.txt', '1_295_Dataset.txt', '1_296_Dataset.txt', '1_297_Dataset.txt', '1_298_Dataset.txt', '1_299_Dataset.txt', '1_29_Dataset.txt', '1_2_Dataset.txt', '1_300_Dataset.txt', '1_301_Dataset.txt', '1_302_Dataset.txt', '1_303_Dataset.txt', '1_304_Dataset.txt', '1_305_Dataset.txt', '1_306_Dataset.txt', '1_307_Dataset.txt', '1_308_Dataset.txt', '1_309_Dataset.txt', '1_30_Dataset.txt', '1_310_Dataset.txt', '1_311_Dataset.txt', '1_312_Dataset.txt', '1_313_Dataset.txt', '1_314_Dataset.txt', '1_315_Dataset.txt', '1_316_Dataset.txt', '1_317_Dataset.txt', '1_318_Dataset.txt', '1_319_Dataset.txt', '1_31_Dataset.txt', '1_320_Dataset.txt', '1_321_Dataset.txt', '1_322_Dataset.txt', '1_323_Dataset.txt', '1_324_Dataset.txt', '1_325_Dataset.txt', '1_326_Dataset.txt', '1_327_Dataset.txt', '1_328_Dataset.txt', '1_329_Dataset.txt', '1_32_Dataset.txt', '1_330_Dataset.txt', '1_331_Dataset.txt', '1_332_Dataset.txt', '1_333_Dataset.txt', '1_334_Dataset.txt', '1_335_Dataset.txt', '1_336_Dataset.txt', '1_337_Dataset.txt', '1_338_Dataset.txt', '1_339_Dataset.txt', '1_33_Dataset.txt', '1_340_Dataset.txt', '1_341_Dataset.txt', '1_342_Dataset.txt', '1_343_Dataset.txt', '1_344_Dataset.txt', '1_345_Dataset.txt', '1_346_Dataset.txt', '1_347_Dataset.txt', '1_348_Dataset.txt', '1_349_Dataset.txt', '1_34_Dataset.txt', '1_350_Dataset.txt', '1_351_Dataset.txt', '1_352_Dataset.txt', '1_353_Dataset.txt', '1_354_Dataset.txt', '1_355_Dataset.txt', '1_356_Dataset.txt', '1_357_Dataset.txt', '1_358_Dataset.txt', '1_359_Dataset.txt', '1_35_Dataset.txt', '1_360_Dataset.txt', '1_361_Dataset.txt', '1_362_Dataset.txt', '1_363_Dataset.txt', '1_364_Dataset.txt', '1_365_Dataset.txt', '1_366_Dataset.txt', '1_367_Dataset.txt', '1_368_Dataset.txt', '1_369_Dataset.txt', '1_36_Dataset.txt', '1_370_Dataset.txt', '1_371_Dataset.txt', '1_372_Dataset.txt', '1_373_Dataset.txt', '1_374_Dataset.txt', '1_375_Dataset.txt', '1_376_Dataset.txt', '1_377_Dataset.txt', '1_378_Dataset.txt', '1_379_Dataset.txt', '1_37_Dataset.txt', '1_380_Dataset.txt', '1_381_Dataset.txt', '1_382_Dataset.txt', '1_383_Dataset.txt', '1_384_Dataset.txt', '1_385_Dataset.txt', '1_386_Dataset.txt', '1_387_Dataset.txt', '1_388_Dataset.txt', '1_389_Dataset.txt', '1_38_Dataset.txt', '1_390_Dataset.txt', '1_391_Dataset.txt', '1_392_Dataset.txt', '1_393_Dataset.txt', '1_394_Dataset.txt', '1_395_Dataset.txt', '1_396_Dataset.txt', '1_397_Dataset.txt', '1_398_Dataset.txt', '1_399_Dataset.txt', '1_39_Dataset.txt', '1_3_Dataset.txt', '1_400_Dataset.txt', '1_401_Dataset.txt', '1_402_Dataset.txt', '1_403_Dataset.txt', '1_404_Dataset.txt', '1_405_Dataset.txt', '1_406_Dataset.txt', '1_407_Dataset.txt', '1_408_Dataset.txt', '1_409_Dataset.txt', '1_40_Dataset.txt', '1_410_Dataset.txt', '1_411_Dataset.txt', '1_412_Dataset.txt', '1_413_Dataset.txt', '1_414_Dataset.txt', '1_415_Dataset.txt', '1_416_Dataset.txt', '1_417_Dataset.txt', '1_418_Dataset.txt', '1_419_Dataset.txt', '1_41_Dataset.txt', '1_420_Dataset.txt', '1_421_Dataset.txt', '1_422_Dataset.txt', '1_423_Dataset.txt', '1_424_Dataset.txt', '1_425_Dataset.txt', '1_42_Dataset.txt', '1_43_Dataset.txt', '1_44_Dataset.txt', '1_45_Dataset.txt', '1_46_Dataset.txt', '1_47_Dataset.txt', '1_48_Dataset.txt', '1_49_Dataset.txt', '1_4_Dataset.txt', '1_50_Dataset.txt', '1_51_Dataset.txt', '1_52_Dataset.txt', '1_53_Dataset.txt', '1_54_Dataset.txt', '1_55_Dataset.txt', '1_56_Dataset.txt', '1_57_Dataset.txt', '1_58_Dataset.txt', '1_59_Dataset.txt', '1_5_Dataset.txt', '1_60_Dataset.txt', '1_61_Dataset.txt', '1_62_Dataset.txt', '1_63_Dataset.txt', '1_64_Dataset.txt', '1_65_Dataset.txt', '1_66_Dataset.txt', '1_67_Dataset.txt', '1_68_Dataset.txt', '1_69_Dataset.txt', '1_6_Dataset.txt', '1_70_Dataset.txt', '1_71_Dataset.txt', '1_72_Dataset.txt', '1_73_Dataset.txt', '1_74_Dataset.txt', '1_75_Dataset.txt', '1_76_Dataset.txt', '1_77_Dataset.txt', '1_78_Dataset.txt', '1_79_Dataset.txt', '1_7_Dataset.txt', '1_80_Dataset.txt', '1_81_Dataset.txt', '1_82_Dataset.txt', '1_83_Dataset.txt', '1_84_Dataset.txt', '1_85_Dataset.txt', '1_86_Dataset.txt', '1_87_Dataset.txt', '1_88_Dataset.txt', '1_89_Dataset.txt', '1_8_Dataset.txt', '1_90_Dataset.txt', '1_91_Dataset.txt', '1_92_Dataset.txt', '1_93_Dataset.txt', '1_94_Dataset.txt', '1_95_Dataset.txt', '1_96_Dataset.txt', '1_97_Dataset.txt', '1_98_Dataset.txt', '1_99_Dataset.txt', '1_9_Dataset.txt']

trainfiles = allfiles[:20]
valfiles = allfiles[:30]
testfiles = allfiles[:40]


COMPRESSED = False
dir = "data/TTBar_35_PU"

def create_numpy_data(files):
    all_data = []
    for fname in files:
        print("processsing: " + fname)
        #with gzip.open(dir + "/" + fname, 'rb') as f:
        data = np.genfromtxt(dir + "/" + fname, delimiter='\t', dtype=np.float32)
        all_data.append(data)
    data = np.vstack(all_data)
    return data


fnames = os.listdir(dir)
n_files = len(fnames)
i_train = int(n_files * 0.5)
i_val = int(n_files * 0.75)

data_train = create_numpy_data(trainfiles)#create_numpy_data(fnames[:i_train])
data_val = create_numpy_data(valfiles)#create_numpy_data(fnames[i_train:i_val])
data_test = create_numpy_data(testfiles)#create_numpy_data(fnames[i_val:])
data_debug = data_train[:500, :]


def balance_data(data, max_ratio=0.5, verbose=True):
    """ Balance the data. """ 
    df = pd.DataFrame(data, columns=dataset.datalabs)
    data_neg = df[df[dataset.target_lab] == 0.0]
    data_pos = df[df[dataset.target_lab] != 0.0]

    n_pos = data_pos.shape[0]
    n_neg = data_neg.shape[0]
    if verbose:
        print("Number of negatives: " + str(n_neg))
        print("Number of positive: " + str(n_pos))
        print("ratio: " + str(n_neg / n_pos))
    
    data_neg = data_neg.sample(n_pos)
    balanced_data = pd.concat([data_neg, data_pos])
    balanced_data = balanced_data.sample(frac=1)  # Shuffle the dataset
    return balanced_data


if COMPRESSED:
    np.savez_compressed('data/train.npz', data_train)
    np.savez_compressed('data/val.npz', data_val)
    np.savez_compressed('data/test.npz', data_test)
    np.savez_compressed('data/debug.npz', data_debug)
else:    
    np.save('data/train.npy', balance_data(data_train))
    np.save('data/val.npy', balance_data(data_val))
    np.save('data/test.npy', data_test)
    np.save('data/debug.npy', data_debug)

    