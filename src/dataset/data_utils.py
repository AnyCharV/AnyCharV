import numpy as np
from PIL import Image
import cv2
import bezier
import torchvision.transforms as transforms


def create_mask(image, bbox):
    width, height = image.size
    x1, y1, x2, y2 = bbox
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def crop_fg_keep_ratio(image, ref_bbox, tgt_bbox):
    ref_h = ref_bbox[3] - ref_bbox[1]
    ref_w = ref_bbox[2] - ref_bbox[0]
    tgt_h = tgt_bbox[3] - tgt_bbox[1]
    tgt_w = tgt_bbox[2] - tgt_bbox[0]

    if ref_h > ref_w:
        ref_w_new = ref_h * tgt_w / tgt_h
        image = transforms.functional.crop(image, ref_bbox[1], ref_bbox[0] - (ref_w_new - ref_w) / 2, ref_bbox[3] - ref_bbox[1], ref_w_new)
    else:
        ref_h_new = ref_w * tgt_h / tgt_w
        image = transforms.functional.crop(image, ref_bbox[1] - (ref_h_new - ref_h) / 2, ref_bbox[0], ref_h_new, ref_bbox[2] - ref_bbox[0])

    
    # ref_bbox[2] = ref_bbox[2] + (ref_w_new - ref_w) / 2
    # ref_bbox[0] = ref_bbox[0] - (ref_w_new - ref_w) / 2
    # image = transforms.functional.crop(image, ref_bbox[1], ref_bbox[0], ref_bbox[3] - ref_bbox[1], ref_bbox[2] - ref_bbox[0])

    return image


def crop_fg(image, ref_bbox):
    image = transforms.functional.crop(image, ref_bbox[1], ref_bbox[0], ref_bbox[3] - ref_bbox[1], ref_bbox[2] - ref_bbox[0])
    return image


def mask_image(image, mask, reverse=False):
    if type(mask) == Image.Image:
        mask = np.array(mask) == 255
    if reverse:
        mask = ~mask
    image_array = np.array(image)
    masked_image_array = np.where(mask[:, :, np.newaxis], image_array, 0)
    masked_image = Image.fromarray(masked_image_array)
    return masked_image


def get_seg_pil_from_result(result, shape):
    polygons = [result['segments']['x'], result['segments']['y']]
    polygons = np.array(polygons).T
    polygons = polygons * shape
    polygons = polygons.astype(np.int32)   
    mask = np.zeros(shape[::-1], dtype=np.uint8)
    cv2.fillPoly(mask, [polygons], color=(255, 255, 255))
    mask_pil = Image.fromarray(mask, mode="L")
    return mask_pil, polygons


def get_box_pil_from_result(result, shape):
    x1, y1, x2, y2 = result['box']['x1'], result['box']['y1'], result['box']['x2'], result['box']['y2']
    width, height = shape
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)
    polygons = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]])
    mask = np.zeros(shape[::-1], dtype=np.uint8)
    cv2.fillPoly(mask, [polygons], color=(255, 255, 255))
    mask_pil = Image.fromarray(mask, mode="L")
    return mask_pil, (x1, y1, x2, y2)


def random_box(tgt_bbox, shape, sample_num=20, random_width=0.2):
    top_nodes = np.asfortranarray([
            [tgt_bbox[0],(tgt_bbox[0]+tgt_bbox[2])/2 , tgt_bbox[2]],
            [tgt_bbox[1], tgt_bbox[1], tgt_bbox[1]],
        ])
    down_nodes = np.asfortranarray([
            [tgt_bbox[2],(tgt_bbox[0]+tgt_bbox[2])/2 , tgt_bbox[0]],
            [tgt_bbox[3], tgt_bbox[3], tgt_bbox[3]],
        ])
    left_nodes = np.asfortranarray([
            [tgt_bbox[0],tgt_bbox[0] , tgt_bbox[0]],
            [tgt_bbox[3], (tgt_bbox[1]+tgt_bbox[3])/2, tgt_bbox[1]],
        ])
    right_nodes = np.asfortranarray([
            [tgt_bbox[2],tgt_bbox[2] , tgt_bbox[2]],
            [tgt_bbox[1], (tgt_bbox[1]+tgt_bbox[3])/2, tgt_bbox[3]],
        ])

    top_curve = bezier.Curve(top_nodes,degree=2)
    right_curve = bezier.Curve(right_nodes,degree=2)
    down_curve = bezier.Curve(down_nodes,degree=2)
    left_curve = bezier.Curve(left_nodes,degree=2)
    curve_list=[top_curve,right_curve,down_curve,left_curve]

    interp_num=sample_num*4
    random_point = np.random.rand(sample_num)
    random_point = np.concatenate([random_point, random_point[0:1]])
    x = np.linspace(0, len(random_point) - 1, interp_num+1)
    scale_array = np.interp(x, np.arange(len(random_point)), random_point)[:-1]

    pt_list=[]
    x_c = (tgt_bbox[0]+tgt_bbox[2])/2
    y_c = (tgt_bbox[1]+tgt_bbox[3])/2
    for j, curve in enumerate(curve_list):
        for i in range(1,sample_num-1):
            sample_x = curve.evaluate(i*(1/sample_num))[0][0]
            sample_y = curve.evaluate(i*(1/sample_num))[1][0]
            pt_list.append(
                (sample_x-scale_array[j*sample_num+i]*(sample_x-x_c)*random_width,
                sample_y-scale_array[j*sample_num+i]*(sample_y-y_c)*random_width,))

    mask = np.zeros(shape[::-1], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pt_list).astype(np.int32)], color=(255, 255, 255))
    mask_pil = Image.fromarray(mask, mode="L")
    return mask_pil


def random_seg(polygons, shape, sample_num=5, random_ratio=20):
    ori_mask = np.zeros(shape[::-1], dtype=np.uint8)
    cv2.fillPoly(ori_mask, [polygons], color=(255, 255, 255))

    random_point = np.random.randn(sample_num)
    random_point = np.concatenate([random_point, random_point[0:1]])
    x = np.linspace(0, len(random_point) - 1, len(polygons)+1)
    scale_array = np.interp(x, np.arange(len(random_point)), random_point)[:-1]
    polygons[:,0] = polygons[:,0] + scale_array * shape[0] / random_ratio
    random_point = np.random.randn(sample_num)
    random_point = np.concatenate([random_point, random_point[0:1]])
    x = np.linspace(0, len(random_point) - 1, len(polygons)+1)
    scale_array = np.interp(x, np.arange(len(random_point)), random_point)[:-1]
    polygons[:,1] = polygons[:,1] + scale_array * shape[1] / random_ratio
    mask = np.zeros(shape[::-1], dtype=np.uint8)
    cv2.fillPoly(mask, [polygons], color=(255, 255, 255))

    mask = np.where(mask==255, 255, ori_mask)
    mask_pil = Image.fromarray(mask, mode="L")
    return mask_pil
