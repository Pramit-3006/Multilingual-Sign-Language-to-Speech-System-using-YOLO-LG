import json

def load_class_names(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mp = json.load(f)
    # preserve order
    classes = [entry['label'] for entry in mp['classes']]
    label_to_text = {entry['label']: entry['text'] for entry in mp['classes']}
    return classes, label_to_text
