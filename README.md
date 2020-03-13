# novelty-detection
Seeing how CNNs respond to unknown classes

## Data
### in_out_class.csv
This is hand-annotated data from iNaturalist. The most up-to-date version can be found [here](https://docs.google.com/spreadsheets/d/1ZbmtlW-vzdHBqO0ZceAwVUiLEFYpZdytlRBlmeXLle4/edit?usp=sharing "in_out_class.csv")
The data taken directly from iNaturalist includes the biological groups and scientific names of natural things. Annotators included the common English name(s) for each creature, their relation to ImageNet, any relevant notes, and their initials. For details regarding annotation guidelines, see [this link](https://docs.google.com/document/d/1YBKqKgjwUQ-o9IMifPO8xORE6UxroQs-0jwkAGuo3hU/edit?usp=sharing).

### alexnet_inat_results/ 
#### inat_results_top_choice.json
This json file contains the results from testing a pre-trained AlexNet (trained on ImageNet) on images from iNaturalist. It only includes the top one result (i.e. the label chosen by the network) for each image in iNaturalist, and so is most efficient when looking into the distribution of labels chosen for a certain type of creature.

#### Biological group files
Each of these folders contains all of the results of testing a pre-trained AlexNet (trained on ImageNet) on images from iNaturalist in the given biological group. This includes all possible labels, their scores, and their confidence values for each image. Since ImageNet has 1000 classes, that means that each image in iNaturalist has 3 vectors of length 1000 to store the label, score, and confidence value information. Each of the files within these folders contains the data for a single species within the given biological group

## Code
### class_in_or_out.py
This script plots the distribution of the top n CNN labels for all (or part) of the image data. Looking at all species of interest, it averages the frequency of the top n labels. Note that the top n labels are not necessarily in the same order for each species, and so the labels themselves are ignored. 

The species each fall under one of four annotated ImageNet relationship categories: in ImageNet, not in ImageNet, parent in ImageNet, and relative in Imagenet. These annotations are taken from ```in_out_class.csv```. The plots may be stratified by these relationship categories.

As an example, this code can plot the frequency of the top 10 labels over all bird images, and split by the species' relationship to Imagenet. The resulting plot will show the average distribution of label frequencies. The top label frequency, for example, is the frequency of the top occuring label over all images averaged over a given species, regardless of what that top label actually was.

This plot shows the frequency of the top 20 labels over all bird species in iNaturalist:

![Bird Label Frequencies](https://github.com/noameshed/novelty-detection/blob/master/top_20_aves.png)

### plot_result_distribution.py
This script plots the distribution of CNN labels over each species. It does so by counting the number of occurrences of each label over many images of that species and normalizing the result to get a frequency distribution rather than an occurrence count distribution. There is an option to color and label each point according to the average confidence of the label. This can help us understand what common mistakes the network makes when classifying images of a given species.

In this example plot, we can see the distribution of all labels guessed by the network in the set of African Penguin images. It shows that approximately 19% of the images are classified as magpie, 19% as goose, etc. Interestingly, the king_penguin label is only awarded to 5% of the images and is tied for the 5th most common label.

![African Penguin Distribution](https://github.com/noameshed/novelty-detection/blob/master/Spheniscus_demersus.png)

### alexnet_novelty.py
This script tests AlexNet (pretrained on ImageNet) on all of the data from iNaturalist and saves the result into the ```alexnet_inat_results/``` folder.
