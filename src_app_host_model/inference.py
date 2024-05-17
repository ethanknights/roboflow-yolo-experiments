from ultralytics import YOLO


def inference_yolo(model_file_name, image_path):

    model = YOLO(model_file_name)
    results = model(image_path, conf = 0.5, iou = 0.6)
    r = results[0]

    if r.probs.top1conf > 0.5:
        top1_category_str = (r.names[r.probs.top1])
    else:
        top1_category_str = 'Unknown (Low confidence)'

    probabilities = [format(r.probs.data[i].item(), '.4f') for i in r.probs.top5]
    dict_probabilities = {r.names[i]: prob for i, prob in zip(r.probs.top5, probabilities)}

    print(dict_probabilities)

    return top1_category_str, dict_probabilities
