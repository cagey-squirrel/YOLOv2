from PIL import ImageDraw
import torchvision
from IPython.display import display
import torch
from matplotlib import pyplot as plt
import matplotlib

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

def show_images_with_boxes(input_tensor, output_tensor):
    to_img = torchvision.transforms.ToPILImage()
    for img, predictions in zip(input_tensor, output_tensor):
        img = to_img(img)
        if 0 in predictions.shape: # empty tensor
            display(img)
            continue

        # ... dodje do nizova koji sadrze elemente
        # iz svakog elementarnog niza uzmi cetvrti element (sigurnost da postoji objekat)
        confidences = predictions[..., 4].flatten()
        
        # Od svih elementarnih nizova uzmi prva 4 elementa (koord i velicinu)
        # contiguous radi poravnanje u memoriji
        # view(-1, 4) -> -1 radi flatten a 4 ga dodatno deli na nizove od po 4 elementa
        # Dobija se matrica br_predvidjanja x 4
        boxes = (
            predictions[..., :4].contiguous().view(-1, 4)
        )  # only take first four features: x0, y0, w, h

        # Uzima samo verovatnoce klasa iz predikcija
        # Daje matricu br_predvidjanja x br_klasa
        classes = predictions[..., 5:].contiguous().view(boxes.shape[0], -1)

        # Prvi i treci element pomnozi sirinom slike
        boxes[:, ::2] *= img.width

        # Drugi i cetvrti element pomnozi visinom slike
        boxes[:, 1::2] *= img.height

        # Gornji levi ugao kvadrata je ulevo i gore za pola kocke
        # A donji desni ugao je dole i udesno za pola kocke
        boxes = (torch.stack([
                    boxes[:, 0] - boxes[:, 2] / 2,
                    boxes[:, 1] - boxes[:, 3] / 2,
                    boxes[:, 0] + boxes[:, 2] / 2,
                    boxes[:, 1] + boxes[:, 3] / 2,
        ], -1, ).cpu().to(torch.int32).numpy())


        for box, confidence, class_ in zip(boxes, confidences, classes):
            if confidence < 0.01:
                print("low conf")
                continue # don't show boxes with very low confidence
            # make sure the box fits within the picture:
            box = [
                max(0, int(box[0])),
                max(0, int(box[1])),
                min(img.width - 1, int(box[2])),
                min(img.height - 1, int(box[3])),
            ]
            
            try:  # either the class is given as the sixth feature
                idx = int(class_.item())
            except ValueError:  # or the 20 softmax probabilities are given as features 6-25
                idx = int(torch.max(class_, 0)[1].item())
            try:
                class_ = CLASSES[idx]  # the first index of torch.max is the argmax.
            except IndexError: # if the class index does not exist, don't draw anything:
                print("no class")
                continue

            
            color = (  # green color when confident, red color when not confident.
                int((1 - (confidence.item())**0.8 ) * 255),
                int((confidence.item())**0.8 * 255),
                0,
            )
            draw = ImageDraw.Draw(img)
            draw.rectangle(box, outline=color)
            draw.text(box[:2], class_, fill=color)

            f,ax = plt.subplots()
            rect = matplotlib.patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1])
            ax.imshow(img)
            ax.add_patch(rect)

            plt.show()
        display(img)