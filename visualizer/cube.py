# %%
import os
print(os.getcwd())
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    files = os.listdir(folder)
    files = [f for f in files if f.endswith('.bmp')]
    files.sort(key=lambda x: int(x.split('.')[0].split('nm')[0].split('_')[-1]))
    files.reverse()
    for filename in files:
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    print('min:', min([int(f.split('.')[0].split('nm')[0].split('_')[-1]) for f in files]))
    print('max:', max([int(f.split('.')[0].split('nm')[0].split('_')[-1]) for f in files]))
    return images

def make(str):
    name = str
    hsi_images = load_images_from_folder(name)
    images = []

    # Generate noise images
    shape = (512, 640)

    for i in range(len(hsi_images)):
        img = hsi_images[i]
        img = img.resize(shape)
        images.append(img)

    scale = 1
    leng = len(images) * scale
    from tqdm import tqdm
    x_size = images[0].size[0]
    y_size = images[0].size[1]
    import matplotlib.cm as cm
    for idx, img in enumerate(tqdm(images, desc="(1/3) Make cube images")):
        iid = idx + 1
        # img : [0,1] -> [0,255]
        img = np.array(img)
        img = img * 255
        img = Image.fromarray(img)
        plt.imshow(img, extent=[leng - iid * scale, leng - iid * scale + img.size[0], 
                                leng - iid * scale, leng - iid * scale + img.size[1]], cmap=cm.gray)
        
    plt.xlim(0, leng + x_size)
    plt.ylim(0, leng + y_size)
    plt.axis('off')
    print(f'Saving image to gen_images/{name.split("/")[-1]}.png')
    plt.savefig('gen_images/' + name.split('/')[-1] + '.png')
    plt.close()

    result = []
    x_size = images[0].size[0]
    y_size = images[0].size[1]
    for i in tqdm(range(len(images)), desc="(2/3) Make gif images"):
        # Plot all images in a cube shape
        # get permutation of images
        ima = images[:i]
        for idx, img in enumerate(ima):
            iid = idx + 1
            plt.imshow(img, extent=[leng - iid * scale, leng - iid * scale + img.size[0], 
                                    leng - iid * scale, leng - iid * scale + img.size[1]], cmap=cm.gray)

        # front image
        # front = Image.open('KSC.png')
        # front = front.resize((int(front.size[0]), int(front.size[1])))
        # plt.imshow(front) 
        plt.xlim(0, leng + x_size)
        plt.ylim(0, leng + y_size)
        plt.axis('off')
        result.append(plt.gcf())
        plt.close()


    import imageio

    # Save results as a GIF
    frames = []

    for fig in tqdm(result, desc="(3/3) Saving GIF"):
        # Save the current figure as a temporary image
        temp_path = 'gen_images/temp_image.png'
        fig.savefig(temp_path, format='png')
        # Read the temporary image into frames
        frame = imageio.v3.imread(temp_path)
        frames.append(frame)
    # Save frames as a GIF

    gif_path_adjusted = 'gen_images/' + name.split('/')[-1] + '.gif'
    imageio.mimsave(gif_path_adjusted, frames, format='GIF', fps=10000)

    # Clean up temporary files
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print(f'GIF saved to {gif_path_adjusted}')

# %%
# make('/app/results/epoch_5000/water_cropped')
make('/app/results/epoch_5000/plastic_cropped')
# make('/app/results/epoch_9600/plastic_cropped')
# %%
