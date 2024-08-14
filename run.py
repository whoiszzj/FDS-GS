import os

exp_name = 'FDS-GS'

dataset = 'MipNeRF360'
scan_list = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]

data_dir = './data/{}'.format(dataset)
output_dir = './output/{}'.format(dataset)
scan_list.sort()
print("scan_list: ", scan_list)
for scan in scan_list:
    print("Training on scan %s" % scan)
    print('python train.py -s %s/%s -m %s/%s/%s --eval' % (data_dir, scan, output_dir, scan, exp_name))
    os.system('python train.py -s %s/%s -m %s/%s/%s --eval' % (data_dir, scan, output_dir, scan, exp_name))
for scan in scan_list:
    print("Rendering on scan %s" % scan)
    print('python render.py -m %s/%s/%s --iteration 30000 --skip_train' % (output_dir, scan, exp_name))
    os.system('python render.py -m %s/%s/%s --iteration 30000  --skip_train' % (output_dir, scan, exp_name))
for scan in scan_list:
    print("Evaluating on scan %s" % scan)
    print('python metrics.py -m %s/%s/%s' % (output_dir, scan, exp_name))
    os.system('python metrics.py -m %s/%s/%s' % (output_dir, scan, exp_name))
print("Done!")
