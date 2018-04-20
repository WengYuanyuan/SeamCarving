
import cv2
import numpy as np
import seamOperation as sop
import boundingbox




def RemoveObject(img, mask_remove, mask_protect):
    # mask_remove and mask_protect are [0, 255] binary map
    # mask_remove = mask_remove - mask_protect
    # convert mask_remove into [0 ,1] binary map, and remove the overlap region from mask_remove
    # mask_remove[mask_remove > 0] = 1
    # mask_remove[mask_remove != 1] = 0
    # # convert mask_protect into [0 ,1] binary map
    # mask_protect[mask_remove > 0] = 1

    # generate bounding box
    bb_w0, bb_h0, bb_ww, bb_hh = boundingbox.findRect(mask_remove)
    dh = bb_hh - bb_h0
    dw = bb_ww - bb_w0
    step = dh + dw

    
    while step > 0:
        # remove vertical seam if dh > dw
        if dh <= dw:

            energy = sop.GradientEnergyMap(img)
            energy = energy + (0 * mask_protect)
            energy = energy + (-1000 * mask_remove)

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
        print step
        if step == 158:

            cv2.imwrite('remove_initial_energy.jpg', mask_remove)

    return img

def seams_Marker(image, seams, mark_red=(255, 0, 0)):
    """
        input: image and inserted seams
        output: image with marking a series seams.
    """
    xi = np.arange(image.shape[0])
    # mark the image seam by seam
    for s in seams:
        image[xi,s] = mark_red
        
        
if __name__ == "__main__":
    """
    comment here
    """
    image = cv2.imread('set1.jpg')
    mask_remove = cv2.imread('mask_r2.jpg',0)
    #mask_remove[mask_remove != 0]=255
    
    # mask_protect = cv2.imread('protect.jpg',0)
    mask_protect = np.zeros(image.shape[:2])

    # for i in range(mask_remove.shape[0]):
    #     print mask_remove[i]
    # mask_protect = np.zeros(image.shape[:2])

    image_remove = RemoveObject(image, mask_remove, mask_protect)

    cv2.imwrite('remove_done.jpg', image_remove)














