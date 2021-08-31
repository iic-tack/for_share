import pandas as pd
import numpy as np
import glob
import cv2

images = glob.glob("hogehoge")
labels = glob.glob("fugafuga")
out_path = "piyopiyo"

def eval(target_img, gt_img):
    tmp = (target_img > 0) * 1 + (gt_img > 0) * 2
    
    tp = np.count_nonzero(tmp==3)
    fn = np.count_nonzero(tmp==2)
    fp = np.count_nonzero(tmp==1)
    tn = np.count_nonzero(tmp==0)
    
    acc = (tp + tn) / (tp + fn + fp + tn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    f = 2*prec*rec/(prec+rec)
    iou = tp/(tp+fp+fn)
    
    return acc, prec, rec, f, iou

rec = pd.DataFrame(columns=["image", "acc", "prec", "recall", "f1", "IoU"])

for i in range(len(images)):
    im_tmp = cv2.imread(images[i])
    lb_tmp = cv2.imread(labels[i])
    
    acc, prec, recall, f, iou = eval(im_tmp, lb_tmp)
    rec.loc[str(i).zfill(3)] = [f"img{str(i).zfill(3)}", acc, prec, recall, f, iou]

rec.loc["average score"] = ["score", rec["acc"].mean(), rec["prec"].mean(), rec["recall"].mean(), rec["f1"].mean(), rec["IoU"].mean()]

rec.to_csv(out_path, index=False)
