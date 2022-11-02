## Importing Libraries
import os
import sys
import pyrootutils

root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)

import shap
import timm
import torch
import urllib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T


from PIL import Image
from captum.robust import PGD, FGSM
from matplotlib.colors import LinearSegmentedColormap
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from captum.attr import (
    DeepLift,
    Saliency,
    Occlusion,
    NoiseTunnel,
    GradientShap,
    IntegratedGradients,
    visualization as viz,
)


device = torch.device("cuda")


model = timm.create_model("vit_base_patch32_224", pretrained=True)
model.eval()
model = model.to(device)


# Download human-readable labels for ImageNet.
# get the classnames
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
)
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

image_list = os.listdir("s7/10_pics/")

print("image_list: ", image_list)

for image in image_list:

    print("Processing image: ", image)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    transform_normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    img = Image.open("s7/10_pics/" + image)

    transformed_img = transform(img)

    img_tensor = transform_normalize(transformed_img)
    img_tensor = img_tensor.unsqueeze(0)

    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]
    # print("Predicted:", predicted_label, "(", prediction_score.squeeze().item(), ")")

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(
        img_tensor, target=pred_label_idx, n_steps=200
    )

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )

    sth = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        cmap=default_cmap,
        show_colorbar=True,
        sign="positive",
        outlier_perc=1,
        use_pyplot=False,
    )

    sth[0].savefig('s7/output/' + image + "_ig.png")

    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(
        img_tensor, nt_samples=10, nt_type="smoothgrad_sq", target=pred_label_idx
    )
    sth = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False,
    )
    sth[0].savefig('s7/output/' + image + "_ig_with_noisetunnel.png")

    torch.manual_seed(0)
    np.random.seed(0)

    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([img_tensor * 0, img_tensor * 1])

    attributions_gs = gradient_shap.attribute(
        img_tensor,
        n_samples=50,
        stdevs=0.0001,
        baselines=rand_img_dist,
        target=pred_label_idx,
    )
    sth = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "absolute_value"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False,
    )

    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(
        img_tensor,
        strides=(3, 8, 8),
        target=pred_label_idx,
        sliding_window_shapes=(3, 15, 15),
        baselines=0,
    )

    sth = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
        use_pyplot=False,
    )

    sth[0].savefig('s7/output/' + image + "_occlusion.png")

    # ## SHAP

    # Works well where number of classes are less

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    model_out = model(img_tensor)
    classes = torch.argmax(model_out, axis=1).cpu().numpy()
    # print(f"Classes: {classes}: {np.array(categories)[classes]}")

    img_tensor.shape

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            ),
            T.Lambda(lambda x: x.permute(0, 2, 3, 1)),
        ]
    )

    def predict(imgs: torch.Tensor) -> torch.Tensor:
        imgs = torch.tensor(imgs)
        imgs = imgs.permute(0, 3, 1, 2)

        img_tensor = imgs.to(device)

        output = model(img_tensor)
        return output

    topk = 4
    batch_size = 50
    n_evals = 10000

    # define a masker that is used to mask out partitions of the input image.
    masker_blur = shap.maskers.Image("blur(128,128)", (224, 224, 3))

    # create an explainer with model and image masker
    explainer = shap.Explainer(predict, masker_blur, output_names=categories)

    # feed only one image
    # here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
    # image_np = Image.open("cat.jpeg")
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    shap_values = explainer(
        img_tensor,
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    (shap_values.data.shape, shap_values.values.shape)

    shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]

    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

    shap.image_plot(
        shap_values=shap_values.values,
        pixel_values=shap_values.data,
        labels=shap_values.output_names,
        true_labels=[categories[285]],
        show=False
    )
    # plt.show(block=False)

    plt.savefig('s7/output/' + image + '_shap.png')

    # # ## Captum Robustness

    # transform = T.Compose(
    #     [
    #         T.Resize((224, 224)),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)

    # img_tensor.requires_grad

    saliency = Saliency(model)
    grads = saliency.attribute(img_tensor, target=285)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    original_image = np.transpose(
        (img_tensor.squeeze(0).cpu().detach().numpy() / 2) + 0.5, (1, 2, 0)
    )

    sth = viz.visualize_image_attr(
        None,
        original_image,
        method="original_image",
        title="Original Image",
        use_pyplot=False,
    )

    sth = viz.visualize_image_attr(
        grads,
        original_image,
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Overlayed Gradient Magnitudes",
        use_pyplot=False,
    )

    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input, target=285, **kwargs)

        return tensor_attributions

    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(
        ig, img_tensor, baselines=img_tensor * 0, return_convergence_delta=True
    )
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    # print("Approximation delta: ", abs(delta))

    sth = viz.visualize_image_attr(
        attr_ig,
        original_image,
        method="blended_heat_map",
        sign="all",
        show_colorbar=True,
        title="Overlayed Integrated Gradients",
        use_pyplot=False,
    )

    sth[0].savefig('s7/output/' + image + "_saliency.png")

    # # ## Captum Model Robustness

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            ),
        ]
    )

    def get_prediction(model, image: torch.Tensor):
        model = model.to(device)
        img_tensor = image.to(device)
        with torch.no_grad():
            output = model(img_tensor)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()
        predicted_label = categories[pred_label_idx.item()]

        return predicted_label, prediction_score.squeeze().item()

        # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    # Get original prediction
    pred, score = get_prediction(model, img_tensor)

    def image_show(img, pred):
        npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()
        plt.imshow(npimg)
        plt.title("prediction: %s" % pred)
        plt.show()

    # image_show(img_tensor.cpu(), pred + " " + str(score))

    # Construct FGSM attacker
    fgsm = FGSM(model, lower_bound=-1, upper_bound=1)
    perturbed_image_fgsm = fgsm.perturb(img_tensor, epsilon=0.16, target=285)
    new_pred_fgsm, score_fgsm = get_prediction(model, perturbed_image_fgsm)

    # inv_transform(img_tensor).shape

    # image_show(perturbed_image_fgsm.cpu(), new_pred_fgsm + " " + str(score_fgsm))

    pgd = PGD(
        model,
        torch.nn.CrossEntropyLoss(reduction="none"),
        lower_bound=-1,
        upper_bound=1,
    )  # construct the PGD attacker

    perturbed_image_pgd = pgd.perturb(
        inputs=img_tensor,
        radius=0.13,
        step_size=0.02,
        step_num=7,
        target=torch.tensor([199]).to(device),
        targeted=True,
    )
    new_pred_pgd, score_pgd = get_prediction(model, perturbed_image_pgd)

    # image_show(perturbed_image_pgd.cpu(), new_pred_pgd + " " + str(score_pgd))

    # Feature Ablation

    feature_mask = (
        torch.arange(64 * 7 * 7)
        .reshape(8 * 7, 8 * 7)
        .repeat_interleave(repeats=4, dim=1)
        .repeat_interleave(repeats=4, dim=0)
        .reshape(1, 1, 224, 224)
    )
    # print(feature_mask)

    from captum.attr import FeatureAblation

    model.cpu()
    ablator = FeatureAblation(model)
    attr = ablator.attribute(img_tensor.cpu(), target=285, feature_mask=feature_mask)
    # Choose single channel, all channels have same attribution scores
    pixel_attr = attr[:, 0:1]

    def pixel_dropout(image, dropout_pixels):
        keep_pixels = image[0][0].numel() - int(dropout_pixels)
        vals, sth = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
        return (pixel_attr < vals.item()) * image

    from captum.robust import MinParamPerturbation

    min_pert_attr = MinParamPerturbation(
        forward_func=model,
        attack=pixel_dropout,
        arg_name="dropout_pixels",
        mode="linear",
        arg_min=0,
        arg_max=1024,
        arg_step=16,
        preproc_fn=None,
        apply_before_preproc=True,
    )

    pixel_dropout_im, pixels_dropped = min_pert_attr.evaluate(
        img_tensor.cpu(), target=285, perturbations_per_eval=10
    )
    # print("Minimum Pixels Dropped:", pixels_dropped)

    new_pred_dropout, score_dropout = get_prediction(model, pixel_dropout_im)

    # image_show(pixel_dropout_im.cpu(), new_pred_dropout + " " + str(score_dropout))

    # Grad CAM

    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(281)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = (
        inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    )
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization).save('s7/output/' + image + "_gradcam.jpeg")

    # plt.imshow(visualization)

    from pytorch_grad_cam import GradCAMPlusPlus

    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)

    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = (
        inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    )
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization).save('s7/output/' + image + "_gradcam_plus_plus.jpeg")

    # plt.imshow(visualization)
