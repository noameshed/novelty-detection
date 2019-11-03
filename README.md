# novelty-detection
Seeing how CNNs respond to unknown classes

## Data
### in_out_class.csv
This is hand-annotated data from iNaturalist. The most up-to-date version can be found [here](https://docs.google.com/spreadsheets/d/1ZbmtlW-vzdHBqO0ZceAwVUiLEFYpZdytlRBlmeXLle4/edit?usp=sharing "in_out_class.csv")
The data taken directly from iNaturalist includes the biological groups and scientific names of natural things. Annotators included the common English name(s) for each creature, their relation to ImageNet, any relevant notes, and their initials. For details regarding annotation guidelines, see [this link](https://docs.google.com/document/d/1YBKqKgjwUQ-o9IMifPO8xORE6UxroQs-0jwkAGuo3hU/edit?usp=sharing).

### inat_results_top_choice.json
This json file includes the results of testing a pre-trained AlexNet (trained on ImageNet) on images from iNaturalist. It only includes the top result for each image in iNaturalist, and so is most efficient when looking into the distribution of labels chosen for a certain type of creature.

### inat_results_all_<biological group name>.json
Each of these files stores all of the results of testing a pre-trained AlexNet (trained on ImageNet) on images from iNaturalist within that biological group. It includes all results for each image, including the labels, values, and confidence levels for all possible labels for each image. Since ImageNet has 1000 classes, that means that each image in iNaturalist has 3 vectors of length 1000 to store the label, confidence, and value information.

## Code
### class_in_or_out.py
This script plots the distribution of classes falling under each of four annotated categories: in ImageNet, not in ImageNet, parent in ImageNet, and relative in Imagenet. These annotations are taken from ```in_out_class.csv```.

### alexnet_novelty.py
This script tests AlexNet (pretrained on ImageNet) on all of the data from iNaturalist and saves the result into the ```alexnet_inat_results/``` folder.
