from ultralytics import YOLO
import gradio as gr

# Load trained YOLOv8 model
model = YOLO("weights/best.onnx")


def predict_image(image):
    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    output_image = result.plot()

    # Convert BGR to RGB
    output_image = output_image[:, :, ::-1]

    return output_image


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="YOLOv8 Object Detection",
    description="Upload an image and YOLOv8 will detect objects from the 20 PASCAL VOC classes."
)

demo.launch()