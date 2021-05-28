from PerceptiveInferenceAgent import PerceptiveInferenceAgent
from GenerativeLayer import GenerativeLayer
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime
import pickle
import sys
import os


N_ITER = 1000
IMG_SHAPE = (20, 20)


# def main():
#     agents = create_agent()
#     process = create_process()
#
#     generated_images = []
#     predicted_images = []
#     for t in tqdm(range(N_ITER)):
#         generated_image = sample_image(process, t)
#         predicted_image = predict_image(agents)
#         update_agents(agents, generated_image, predicted_image)
#
#         if t < 10 or t > N_ITER - 10:
#             show_images(generated_image, predicted_image)
#
#         generated_images.append(generated_image)
#         predicted_images.append(predicted_image)
#
#     store_results(generated_images, predicted_images)

def create_agent(prior=None):
    x, y = IMG_SHAPE
    agents = []

    for i in range(x):
        row = []
        for j in range(y):
            row.append(PerceptiveInferenceAgent([200], prior=prior))
        agents.append(row)

    return agents


def create_process():
    slow = GenerativeLayer(parent=None, cycle_time=20*10, amplitude=75, equilibrium=128)
    process = GenerativeLayer(parent=slow, cycle_time=10, amplitude=50)

    return process


def sample_image(process, t):
    x, y = IMG_SHAPE
    if t == 0:
        for t_dummy in range(x*y):
            process.sample(t_dummy)

    img = []
    for i in range(x):
        row = []
        for j in range(y):
            row.append(process.sample(t + i + j*x))
        img.append(row)

    return img


def predict_image(agents):
    x, y = IMG_SHAPE

    img = []
    for i in range(x):
        row = []
        for j in range(y):
            row.append(agents[i][j].predict())
        img.append(row)

    return img


def update_agents(agents, generated_image, predicted_image):
    x, y = IMG_SHAPE

    for i in range(x):
        for j in range(y):
            agents[i][j].update(generated_image[i][j][0][0]["value"], predicted_image[i][j]["layer_contributions"])


def show_images(generated_image, predicted_image):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    x, y = IMG_SHAPE

    for i in range(x):
        for j in range(y):
            generated_image[i][j] = generated_image[i][j][0][0]["value"]
            predicted_image[i][j] = predicted_image[i][j]["value"]

    axs[0].imshow(generated_image, vmin=0, vmax=255, cmap='gray')
    axs[0].set_title("Generated image")
    axs[1].imshow(predicted_image, vmin=0, vmax=255, cmap='gray')
    axs[1].set_title("Predicted image")
    plt.show()


def store_results(generated_images, predicted_images):
    print("Storing results...")

    if not os.path.isdir("results/image_results"):
        os.mkdir("results/image_results")

    file = open(datetime.datetime.now().strftime("results/image_results/%d-%m-%y_%H:%M:%S::%f.results"), "wb")
    pickle.dump((N_ITER, IMG_SHAPE, generated_images, predicted_images), file)
    file.close()

    print("Done!")


"""---------------------------------------------------------------------------------------------------------"""


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        self.progress = tqdm(total=N_ITER)
        self.agents = create_agent(prior={"n": 3, "mu": 128, "sigma": 1})
        self.process = create_process()

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        self.t = np.linspace(0, N_ITER-1, N_ITER)

        self.ax1.imshow(np.zeros(IMG_SHAPE), vmin=0, vmax=255, cmap='gray')
        self.ax1.set_title("Generated image")
        self.ax2.imshow(np.zeros(IMG_SHAPE), vmin=0, vmax=255, cmap='gray')
        self.ax2.set_title("Predicted image")

        animation.TimedAnimation.__init__(self, self.fig, interval=100, blit=True)

    def _draw_frame(self, t):
        self.progress.update(1)

        generated_image = sample_image(self.process, t)
        predicted_image = predict_image(self.agents)
        update_agents(self.agents, generated_image, predicted_image)

        x, y = IMG_SHAPE

        for i in range(x):
            for j in range(y):
                generated_image[i][j] = generated_image[i][j][0][0]["value"]
                predicted_image[i][j] = predicted_image[i][j]["value"]

        self.ax1.imshow(generated_image, vmin=0, vmax=255, cmap='gray')
        self.ax2.imshow(predicted_image, vmin=0, vmax=255, cmap='gray')
        self.fig.suptitle("Iteration: {}".format(t))

    def new_frame_seq(self):
        return iter(range(self.t.size))


vid_name = "vid5.mp4"
if not os.path.isfile('results/image_results/{}'.format(vid_name)):
    print("Rendering animation...", file=sys.stderr)
    ani = SubplotAnimation()
    ani.save('results/image_results/{}'.format(vid_name))
    print("Animation {} saved!".format(vid_name), file=sys.stderr)
else:
    print("{} already exists!".format(vid_name), file=sys.stderr)