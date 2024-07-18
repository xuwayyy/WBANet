import numpy as np
from skimage import io, measure


#  Inputs:  gtImg  = ground truth image
#           tstImg = change map
#  Outputs: FA  = False alarms
#           MA  = Missed alarms
#           OE  = Overall error
#           PCC = Overall accuracy
def evaluate(gtImg, tstImg):
    gtImg[np.where(gtImg > 128)] = 255
    gtImg[np.where(gtImg < 128)] = 0
    tstImg[np.where(tstImg > 128)] = 255
    tstImg[np.where(tstImg < 128)] = 0
    [ylen, xlen] = gtImg.shape
    FA = 0
    MA = 0
    label_0 = np.sum(gtImg == 0)
    label_1 = np.sum(gtImg == 255)
    print(label_0)
    print(label_1)

    for j in range(0, ylen):
        for i in range(0, xlen):
            if gtImg[j, i] == 0 and tstImg[j, i] != 0:
                FA = FA + 1
            if gtImg[j, i] != 0 and tstImg[j, i] == 0:
                MA = MA + 1

    OE = FA + MA
    PCC = 1 - OE / (ylen * xlen)
    PRE = ((label_1 + FA - MA) * label_1 + (label_0 + MA - FA) * label_0) / ((ylen * xlen) * (ylen * xlen))
    KC = (PCC - PRE) / (1 - PRE)
    print(' Change detection results ==>')
    print(' ... ... FP:  ', FA)
    print(' ... ... FN:  ', MA)
    print(' ... ... OE:  ', OE)
    print(' ... ... PCC: ', format(PCC * 100, '.2f'))
    print(' ... ... KC: ', format(KC * 100, '.2f'))


def postprocess1(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    # print(res)
    num = res.max()
    # print(num)
    for i in range(1, num + 1):
        idy, idx = np.where(res == i)
        if len(idy) <= 15:
            res_new[idy, idx] = 0.5
    return res_new


def postprocess(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    # print(res)
    num = res.max()
    # print(num)
    for i in range(1, num + 1):
        idy, idx = np.where(res == i)
        if len(idy) <= 20:
            res_new[idy, idx] = 0
    return res_new