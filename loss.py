import torch

class YoloLoss(torch.nn.Module):

    def __init__(self, lambda_conf_obj_not_detected=1, lambda_class_loss=1, lambda_coord_loss=1):
        super().__init__()
        
        # Loss functions:
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')

        # Lambdas:
        self.lambda_conf_obj_not_detected = lambda_conf_obj_not_detected
        self.lambda_class_loss = lambda_class_loss
        self.lambda_coord_loss = lambda_coord_loss



    def forward(self, x, labels):
        '''
        Calculates Yolo loss for network output x and labels

        Both x and labels have shape: (B, W, H, (5 + Num_Classes)), where B is batch size, W and H are width and height of image measured in number of cells
        Num_Classes is the number of classes present in the dataset



        (5 + Num_Classes):
            - 5 here represents Cx, Cy, Wx, Wy, Conf where:
                - Cx and Cy are coordinates of the center of bounding box. They are measured in units of cells. Ranges from 0 to num_of_cells
                - Wx and Wy are width and height of the bounding box. They are measured in units of anchor size
                    This means that in order to plot bounding box on original image you need to multiply Wx and Wy by anchor size. Both numbers range from 0 to inf
                - Conf is confidence that there is object in this bounding box. Ranges from 0 to 1
            -Num_Classes here represents a one-hot encoded vector which holds the probability that detected object belongs to given class
                For example fifth entry in this vector corresponds to the probability that the object detected belongs to the fifth class
                All values in vector are between 0 and 1 and their sum is equal to 1
        '''
        
        # Label shape: (Cx, Cy, Wx, Wy, Conf, ProbClass1, ProbClass2, ... ProbClassN)

        # Extracting coordinates: these are the first 4 entries of each label: Cx, Cy, Wx, Wy
        predictions_coordinates = x[..., 0:4]
        labels_coordinates = labels[..., 0:4]

        # Extracting confidence: this is the 4th entry of each label: Conf
        prediction_conf = x[..., 4]
        object_present = labels[..., 4] # getting the certainty from labels: 1 for object present, 0 for object absent 
        object_not_present = torch.logical_not(object_present)
        num_objects_present = object_present.sum() + 1
        num_objects_not_present = object_not_present.sum() + 1

        # Extracting class probabilities: these are all entries after the 5th entry: ProbClass1, ProbClass2, ... ProbClassN
        prediction_classes = x[..., 5:]
        labels_classes = labels[..., 5:]

        
        coord_loss = self.lambda_coord_loss * self.mse_loss(object_present[..., None] * predictions_coordinates, labels_coordinates) / num_objects_present
        
        conf_loss_object_detected = self.mse_loss(object_present * prediction_conf, object_present) / num_objects_present
        conf_loss_object_not_detected = self.mse_loss(object_not_present * prediction_conf, object_present) / num_objects_not_present
        conf_loss = conf_loss_object_detected + self.lambda_conf_obj_not_detected * conf_loss_object_not_detected

        class_loss = self.lambda_class_loss * self.ce_loss(object_present[..., None] * prediction_classes, labels_classes) / num_objects_present

        return conf_loss + class_loss + coord_loss
