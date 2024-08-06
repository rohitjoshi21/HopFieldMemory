import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def trainer(vector):
    vector = vector.flatten()
    n = len(vector)
    coefMat = np.zeros((n, n))

    indices = np.arange(n)
    for i in range(n):
        diff_indices = indices - i
        coefMat[i, abs(diff_indices)] = vector[i] * vector[abs(diff_indices)]
    return coefMat


def prediction(curruptedVec, coefMat):
    curruptedVec = curruptedVec.flatten()
    temp = np.dot(coefMat, curruptedVec)  # Matrix-vector multiplication
    predictVec = np.where(temp > 0, 1, -1)  # Binarize prediction based on sign

    return predictVec.reshape(
        (int(np.sqrt(len(predictVec))), int(np.sqrt(len(predictVec))))
    )


def imageGenerator(imageVector):
    cleanImage = np.where(imageVector == 1, 1, -1)  # Binarize input image
    noisyImage = cleanImage + np.random.normal(
        0, 2, cleanImage.shape
    )  # Adding Gaussian noise to the clean image
    noisyImage = np.where(noisyImage >= 0, 1, -1)  # Binarize noisy image
    return cleanImage, noisyImage


def image_to_np(path):
    image = Image.open(path).convert("L").resize((128, 128))  # Resize image to 128x128
    np_vector = np.asarray(image)
    np_vector = np.where(np_vector < 128, -1, 1)  # Binarize image
    return np_vector


# Import the image
image = image_to_np("chess.png")
vector, noisyVec = imageGenerator(image)
coefMatrix = trainer(vector)
predictedVec = prediction(noisyVec, coefMatrix)

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Imported Picture")
plt.subplot(2, 2, 2)
plt.imshow(vector, cmap="gray")
plt.title("Cleaned and Cropped Picture")
plt.subplot(2, 2, 3)
plt.imshow(noisyVec, cmap="gray")
plt.title("Noisy Picture")
plt.subplot(2, 2, 4)
plt.imshow(predictedVec, cmap="gray")

plt.title("Recalled Picture by HNN")
plt.savefig("hopfield.png")
plt.show()
plt.clf()

coefMatrix[:1000, :100].tofile(open("coefMatrix.txt", "w"), sep=" ")

plt.imshow(coefMatrix)
plt.title("Single Coefficient Matrix")
plt.savefig("singleCoeffMatrix")
plt.show()
