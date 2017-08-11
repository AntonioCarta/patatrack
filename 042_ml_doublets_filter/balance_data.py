import os
from dataset import Dataset

remote_data = "/eos/cms/store/cmst3/group/dehep/convPixels/TTBar_13TeV_PU35/"
new_dir = "data/bal_data/"

for f in os.listdir(remote_data):
    if os.path.isfile(new_dir + f):  # balancing already done
        print("Skipping (already done): " + f)
        continue
    try:
        ext = f.split('.')[-1]
        if ext == 'h5':
            print("Loading: " + f)
            train_data = Dataset([remote_data + f]).balance_data()
            train_data.save(new_dir + f)
        elif ext == 'gz':
            print("Loading: " + f)
            with open(remote_data + f, 'rb') as f_zip:
                fc = f_zip.read()
                with open('/data/ml/acarta/tmp.h5', 'wb') as f_new:
                    f_new.write(fc)
            train_data = Dataset(['/data/ml/acarta/tmp.h5']).balance_data()
            train_data.save(new_dir + f[:-3])  # skip .gz from name
        else:  # balancing already done
            print("Skipping (unrecognized extension): " + f)
    except:
        print("Error loading: " + f)
