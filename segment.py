import pixellib
from pixellib.instance import instance_segmentation
segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")
segment_image.segmentImage("C://Users//Ali Traders//Downloads//image_segmentation and detection//image.jpg", output_image_name = "output1.jpg", show_bboxes = True)
