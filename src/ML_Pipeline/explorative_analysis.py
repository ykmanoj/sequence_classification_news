import seaborn as sns
import matplotlib.pyplot as plt

# function to check distribution of labels
from ML_Pipeline.Constants import output_dir, image_dir


def check_dist(dataset,title = "Count Plot"):
  sns.countplot(x='label', data=dataset, palette='hls')
  plt.title(title)
  plt.savefig(image_dir+title+".jpeg")
  plt.clf()