from abc import ABC

from general import *


class DetectBase(ABC):
    @staticmethod
    def forward_handle_input(color_frame, depth_frame=None):
        """
        :param color_frame: rgb彩色视频帧
        :param depth_frame: d深度视频帧
        :return:  rgb彩色视频帧
        """
        return color_frame

    @staticmethod
    def gen_model():
        """
        :param function_nums: 模型权重选择
        :return: 权重模型
        """
        return choose_model()

    @staticmethod
    def detect(color_frame, predictor=None, point=None, bboxes=None):
        """
        :param function_nums: 模型权重选择
        :param color_frame: rgb彩色视频帧
        :param predictor: 权重预测
        :param model: 权重模型
        :param point: 提示点
        :param bboxes: 提示框
        :return: 预测结果
        """
        return get_results(color_frame, predictor=predictor, point=point, bboxes=bboxes)

    @staticmethod
    def delete_model():
        torch.cuda.empty_cache()

    @staticmethod
    def backward_handle_output(results):
        """
        :param results:预测结果
        :return: 预测结果处理
        """
        return results_processing(results)
