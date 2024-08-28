import cv2
import numpy as np
import torch
from ultralytics.models.sam import Predictor as SAMPredictor


def choose_model():
    """
    :param function_nums: 模型权重选择
    :return: 权重模型
    """
    model_weight = 'sam_b.pt'
    predictor = SAMPredictor(overrides=dict(conf=0.25, imgsz=(640, 480), model=model_weight, save=False))
    return predictor


def get_results(color_frame, predictor=None, point=None, bboxes=None):
    """
     :param function_nums: 模型权重选择
     :param color_frame: rgb彩色视频帧
     :param predictor: 权重预测
     :param model: 权重模型
     :param point: 提示点
     :param bboxes: 提示框
     :return: 预测结果
     """
    if point:
        results = predictor(color_frame, points=point, labels=[1])
        return results

    if bboxes:
        results = predictor(color_frame, bboxes=bboxes)
        return results
    else:
        results = predictor(color_frame)
        return results


def results_processing(results):
    """
    :param results:预测结果
    :return: 预测结果处理
    """
    center = (0, 0)
    if results[0].masks is not None:
        for index, contour in enumerate(results[0].masks.xy):
            contour = contour.astype(np.int32)

        rect = cv2.minAreaRect(contour)
        center, wh, angle = rect
        center = list(map(int, center))

    masks = results[0].masks.data.clone()
    if isinstance(masks[0], torch.Tensor):
        masks = np.array(masks.cpu())
    mask = None
    for i, mask in enumerate(masks):
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # masks[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
        mask[mask == 1] = 255
    return center, mask
