import cv2
from collections import deque, defaultdict
from src.config import AppConfig
from src.utils import load_class_names
from src.detector import ONNXYoloDetector, UltralyticsDetector
from src.translator import Translator
from src.tts_engine import TTSEngine

def pick_detector(cfg, class_names):
    if cfg.detector_backend == "onnx":
        return ONNXYoloDetector(cfg.onnx_model_path, class_names, cfg.conf_threshold, cfg.iou_threshold)
    elif cfg.detector_backend == "ultralytics":
        # You can point weights to a YOLOv8/YOLOv5 model fine-tuned on signs
        return UltralyticsDetector("models/yolov8n.pt", class_names, cfg.conf_threshold, cfg.iou_threshold)
    else:
        raise ValueError("Unknown detector backend: %s" % cfg.detector_backend)

def draw_boxes(frame, detections):
    for label, conf, (x1,y1,x2,y2) in detections:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

def main():
    cfg = AppConfig()
    class_names, label_to_text = load_class_names(cfg.mapping_path)
    detector = pick_detector(cfg, class_names)
    translator = Translator(cfg.source_lang, cfg.target_lang, cfg.translation_model)
    tts = TTSEngine(cfg.tts_backend, cfg.voice_rate, cfg.voice_volume)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Change camera_index in config or attach a webcam.")

    # Debounce structures
    recent_labels = deque(maxlen=cfg.hold_frames)
    cooldown = defaultdict(int)
    sentence = []

    print("Press 'SPACE' to speak the current sentence, 'c' to clear, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.infer(frame)
        frame = draw_boxes(frame, detections)

        # simple rule: take the top-1 label per frame, if any
        if detections:
            detections.sort(key=lambda x: x[1], reverse=True)
            label = detections[0][0]
            recent_labels.append(label)

            # manage cooldowns
            for k in list(cooldown.keys()):
                cooldown[k] = max(0, cooldown[k]-1)

            if len(recent_labels) == cfg.hold_frames and len(set(recent_labels)) == 1:
                if cooldown[label] == 0:
                    text = label_to_text.get(label, label)
                    sentence.append(text)
                    cooldown[label] = cfg.cooldown_frames
                    recent_labels.clear()

        # render sentence on frame
        display_text = " ".join(sentence[-cfg.sentence_max_tokens:])
        cv2.rectangle(frame, (10, 15), (10 + min(900, 10*len(display_text)), 50), (0,0,0), -1)
        cv2.putText(frame, display_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if cfg.display:
            cv2.imshow("Sign2Speech", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # translate & speak
            text_in = " ".join(sentence)
            if text_in.strip():
                translated = translator.translate(text_in)
                print(f"[EN] {text_in}\n[TL] {translated}")
                # Use target_lang for TTS if supported by your backend (gTTS supports many langs)
                tts.speak(translated, lang=cfg.target_lang)
                sentence.clear()
        elif key == ord('c'):
            sentence.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
