import torch
import gradio as gr
import torchvision.transforms as T
import torch.nn.functional as F

# from src import utils

# log = utils.get_pylogger(__name__)

def demo():
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # log.info("Running Demo")

    # log.info(f"Instantiating scripted model")
    model = torch.jit.load('model.script.pt')

    # log.info(f"Loaded Model: {model}")

    def recognize_digit(image):
        if image is None:
            return None
        image = T.ToTensor()(image).unsqueeze(0)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        return {label[i]: preds[i] for i in range(10)}

    # im = gr.Image(shape=(32, 32), image_mode="L", invert_colors=True, source="canvas")

    demo = gr.Interface(
        fn=recognize_digit,
        inputs="image",
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(server_name="0.0.0.0", server_port=8080)


if __name__ == "__main__":
    demo()