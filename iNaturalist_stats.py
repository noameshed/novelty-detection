# explore data statistics
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

f = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/'

grp_names = []
grp_count = []
grp_min = np.inf
min_folder = ''
grp_max = 0
max_folder = ''
avg_folder = 0
counter = 0
for i, bio_grp in enumerate(os.listdir(f)):
    class_path = f + bio_grp + '/'
    
    grp_count.append(0)
    for clss in os.listdir(class_path):
        pics = len(os.listdir(class_path + clss + '/'))
        if pics > grp_max:
            grp_max = pics
            max_folder = clss
        if pics < grp_min:
            grp_min = pics
            min_folder = clss
        avg_folder += pics
        grp_count[i] += pics
        counter += 1

    grp_names.append(bio_grp)

avg_folder/=counter

print('smallest folder (%s) has %d images' %(min_folder, grp_min))
print('biggest folder (%s) has %d images' %(max_folder, grp_max))
print('average folder size is %d' %(round(avg_folder)))

'''
Results printed:
smallest folder (Datana ministra) has 14 images
biggest folder (Danaus plexippus) has 3949 images
average folder size is 133
'''

# Plot number of images per class
ax = sns.barplot(grp_names, grp_count)
ax.set_title('Distribution of Images by Biological Group')
ax.set_xlabel('Biological Group')
ax.set_ylabel('Number of images')
for p in ax.patches:
    ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=8, color='black', xytext=(0, 4),
                 textcoords='offset points')
    
plt.show()
