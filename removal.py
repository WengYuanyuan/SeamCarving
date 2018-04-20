
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
    step0 = dh + dw
    step = dh + dw

    
    while step > 0:
        # remove vertical seam
        if dh <= dw:

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
            energy = energy + (100 * mask_protect)
            energy = energy + (-100 * mask_remove)


            seam = sop.findHorSeam(img, energy)

            # update mask_protect and mask_remove
            new_img = sop.removeHorSeam(img, seam)
            mask_remove = sop.removeHorSeam(mask_remove, seam)
            mask_protect = sop.removeHorSeam(mask_protect, seam)

            img = new_img
            dh = dh - 1

        #step = dh + dw
        step -=1
        print step
        if step == step0:

            cv2.imwrite('remove_initial_energy.jpg', mask_remove)

    return img


        
if __name__ == "__main__":
    """
    comment here
    """
    # demenstration set 1
    
    image = cv2.imread('1_input.jpg')
    mask_remove = cv2.imread('1_mask_removal.png',0)
    mask_protect = cv2.imread('1_mask_protection.png',0)

    image_remove = RemoveObject(image, mask_remove, mask_protect)
    cv2.imwrite('1_output.jpg', image_remove)
    
    # demenstration set 2
    image = cv2.imread('2_input.jpg')
    mask_remove = cv2.imread('2_mask_remove.png',0)
    mask_protect = cv2.imread('2_mask_protection_half.png',0)

    image_remove = RemoveObject(image, mask_remove, mask_protect)
    cv2.imwrite('2_output.jpg', image_remove)
    
    
    # demenstration set 3

    image = cv2.imread('3_input.jpg')
    mask_remove = cv2.imread('3_mask_removal.png',0)
    mask_protect = cv2.imread('3_mask_protection.png',0)

    image_remove = RemoveObject(image, mask_remove, mask_protect)
    cv2.imwrite('3_output.jpg', image_remove)













