import os
import xml.etree.ElementTree as ET

# PASCAL VOC classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Class name to class ID
class_to_id = {
    class_name: i
    for i, class_name in enumerate(VOC_CLASSES)
}

def convert_xml_to_yolo(xml_path, output_txt):
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    yolo_labels = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text
        class_id = class_to_id[class_name]

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        # Normalize coordinates
        x_center /= image_width
        y_center /= image_height
        bbox_width /= image_width
        bbox_height /= image_height

        yolo_labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{bbox_width:.6f} {bbox_height:.6f}"
        )

    with open(output_txt, "w") as f:
        f.write("\n".join(yolo_labels))


print("VOC to YOLO conversion function ready!")