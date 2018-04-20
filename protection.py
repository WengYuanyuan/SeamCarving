
import cv2
import numpy as np
import seamOperation as sop
import boundingbox




def ProtectObject(img, hreduce, wreduce, mask_remove, mask_protect):
    # mask_remove and mask_protect are [0, 255] binary map

    dh = hreduce
    dw = wreduce
    step = hreduce + wreduce

    while step > 0:
        # remove vertical seam if dh > dw
        if dh < dw:

            energy = sop.GradientEnergyMap(img)
            energy = energy + (10 * mask_protect)
            energy = energy + (-10 * mask_remove)

            seam = sop.findVerSeam(img, energy)

            # update mask_protect and mask_remove
            new_img = sop.removeVerSeam(img, seam)
            mask_remove = sop.removeVerSeam(mask_remove, seam)
            mask_protect = sop.removeVerSeam(mask_protect, seam)

            img = new_img
            dw = dw - 1

        # remove horizontal seam if dw > dh
        else:
            energy = sop.GradientEnergyMap(img)
            print energy.shape
            energy = energy + (10 * mask_protect)
            energy = energy + (-10 * mask_remove)


            seam = sop.findHorSeam(img, energy)

            # update mask_protect and mask_remove
            new_img = sop.removeHorSeam(img, seam)
            mask_remove = sop.removeHorSeam(mask_remove, seam)
            mask_protect = sop.removeHorSeam(mask_protect, seam)

            img = new_img
            dh = dh - 1

        step = dh + dw

    return img


if __name__ == "__main__":
    """
    comment here
    # """
    image = cv2.imread('set2.jpg')
    mask_protect = cv2.imread('m_all2.png',0)
    #mask_protect[mask_protect != 0]=255
    # for i in range(mask_remove.shape[0]):
    #     print mask_remove[i]
    mask_remove = np.zeros(image.shape[:2])
    hreduce = 0
    wreduce = 384
    
    image_remove = ProtectObject(image, hreduce, wreduce, mask_remove, mask_protect)
    
    cv2.imwrite('protect_done.jpg', image_remove)














