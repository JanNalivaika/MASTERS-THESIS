import glob

from PIL import Image
from natsort import natsorted, ns

def make_gif(frame_folder):

    elems = glob.glob(f"{frame_folder}/*.png")
    elems = natsorted(elems, key=lambda y: y.lower())#
    frames = [Image.open(image) for image in elems]
    frame_one = frames[0]
    frame_one.save("my_awesome_6.gif", format="GIF", append_images=frames,
                   save_all=True, duration=40, loop=0)


if __name__ == "__main__":
    make_gif("robot_rot")