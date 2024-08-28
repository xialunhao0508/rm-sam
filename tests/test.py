import sys

from numpy import ndarray

sys.path.append("..\src")

import cv2
from interface import DetectBase

# 初始化SAM类对象
sam_seg = DetectBase()

# 选择SAM模型
predictor = sam_seg.gen_model()


def test_detect_base(color_img, point):
    # 数据预处理预留接口
    color_frame = sam_seg.forward_handle_input(color_img)

    # 模型推理
    results = sam_seg.detect(color_frame, predictor=predictor, point=point, bboxes=None)

    # 清除模型缓存
    # sam_seg.delete_model()

    # 模型结果后处理
    return sam_seg.backward_handle_output(results)


if __name__ == '__main__':
    # 图片路径
    color_img = cv2.imread('data.jpg')
    point = (366, 251)
    center, mask = test_detect_base(color_img, point)
    # 显示结果
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
