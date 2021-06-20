import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Random state.
RS = 20150101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

for i in range(10):
   a=np.loadtxt('/data4/lzq/tf/GAN/DCGAN/cgan_with_projection/gan_embeding/latent_%d' % i)
   #a=np.load('/data4/lzq/tf/GAN/DCGAN/cgan_with_projection/cgan_embeding/zc%d.npy' % i)
   print(np.shape(a))
   #b=np.loadtxt('/data4/lzq/tf/GAN/DCGAN/cgan_with_projection/cgan_embeding/latent_%d' % i)
   b=i*np.ones(1000)
   a=a[:1000]
   #b=b[:200]
   print(a[1])
   print(b[1])
   if i==0:
      X=a
      y=b
   else:
      X=np.concatenate((X, a), axis=0)
      y=np.concatenate((y, b), axis=0)

#print(np.shape(X))
#print(np.shape(y))

digits_proj = TSNE(random_state=RS).fit_transform(X)

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=5,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

scatter(digits_proj, y)
plt.savefig('digits_tsne-generated.png', dpi=120)
plt.show()
