#!/usr/bin/env python3
""" Write a class Yolo that uses the
Yolo v3 algorithm to perform object detection: """


class Yolo:
    """ class Yolo """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ constructor """
        