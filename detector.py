import os
import numpy as np
import cv2

class BaseDetector:
    def __init__(self, class_names):
        self.class_names = class_names

    def infer(self, frame):
        """Return list of (label, conf, bbox) for each detection where bbox=(x1,y1,x2,y2)."""
        raise NotImplementedError


class ONNXYoloDetector(BaseDetector):
    def __init__(self, model_path, class_names, conf_thres=0.35, iou_thres=0.45):
        super().__init__(class_names)
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # NOTE: adjust to your model's expected input size
        self.img_size = (640, 640)

    def preprocess(self, img):
        img_resized = cv2.resize(img, self.img_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2,0,1))[None, ...]  # NCHW
        return x

    def postprocess(self, preds, orig_shape):
        # This is a generic decoder; adapt to your model's output format.
        # Expected preds shape (N, 84, S) like YOLOv5-ish or custom; adjust if needed.
        # For simplicity, we assume [x,y,w,h,obj,cls...] per anchor.
        boxes = []
        if isinstance(preds, list):
            preds = preds[0]
        preds = np.squeeze(preds)
        if preds.ndim == 1:
            preds = preds[None, :]
        H, W = orig_shape[:2]
        for row in preds:
            if row.shape[0] < 6:
                continue
            obj = row[4]
            cls_scores = row[5:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(obj * cls_scores[cls_id])
            if conf < self.conf_thres:
                continue
            # assume xywh normalized 0..1
            x, y, w, h = row[0], row[1], row[2], row[3]
            x1 = int((x - w/2) * W); y1 = int((y - h/2) * H)
            x2 = int((x + w/2) * W); y2 = int((y + h/2) * H)
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            boxes.append((label, conf, (x1,y1,x2,y2)))
        return boxes

    def infer(self, frame):
        x = self.preprocess(frame)
        out = self.session.run(None, {self.input_name: x})
        return self.postprocess(out, frame.shape)


class UltralyticsDetector(BaseDetector):
    def __init__(self, weights, class_names, conf_thres=0.35, iou_thres=0.45):
        super().__init__(class_names)
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def infer(self, frame):
        results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        dets = []
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                dets.append((label, conf, (x1,y1,x2,y2)))
        return dets
