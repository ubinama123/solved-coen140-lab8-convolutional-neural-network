Download Link: https://assignmentchef.com/product/solved-coen140-lab8-convolutional-neural-network
<br>
<strong>Problem: </strong>Build a convolutional neural network for the recognition task with the fashion MNIST data set. Use the “sparse_categorical_crossentropy” loss function, use ‘adam’ as the optimizer, and train your model for 5 epochs.

Adopt the following convolutional neural network structure:

<ol>

 <li>Input layer</li>

 <li>2d-convolutional layer: filter size 3×3, depth=32, padding=’same’, strides = (1,1), ReLU activation function</li>

 <li>2×2 max pooling layer, strides = (2,2), padding = ‘same’</li>

 <li>2d-convolutional layer: filter size 3×3, depth=64, padding=’same’, strides = (1,1), ReLU activation function</li>

 <li>2×2 max pooling layer, strides = (2,2), padding = ‘same’</li>

 <li>2d-convolutional layer: filter size 3×3, depth=64, padding=’same’, strides = (1,1), ReLU activation function</li>

 <li>Flattening layer</li>

 <li>Fully-connected layer: 64 nodes, ReLU activation function</li>

 <li>(output) Fully-connected layer: 10 nodes, softmax activation function</li>

</ol>




The following code snippet is for your reference:




from tensorflow import keras from tensorflow.keras import layers, models fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() train_images = train_images.reshape((60000, 28, 28, 1)) test_images = test_images.reshape((10000, 28, 28, 1)) # Normalize pixel values to be between 0 and 1 train_images, test_images = train_images / 255.0, test_images / 255.0 model = models.Sequential() model.add(layers.Conv2D(32, (3, 3), activation=’relu’,                          padding=’same’, strides = (1,1), input_shape=(28, 28, 1)))

# … to be completed by yourself







<strong>Include in the report :</strong>

<ol>

 <li>Give the recognition accuracy rate, and show the confusion matrix, both for the test set.</li>

</ol>




<ol start="2">

 <li>For each layer of the network:</li>

</ol>







<ul>

 <li>Manually calculate the number of parameters of that layer. That is, the number of weight elements (include the bias terms). Show you work. Verify whether your results are the same as those given by model.summary().</li>

 <li>Write out the output dimension of that layer.</li>

 <li>Manually calculate the number of multiplications required to generate that layer. The weights have bias terms. You only need to consider the multiplications required to calculate the weighted sums. You don’t need to consider the multiplications involved in the softmax function. Show you work.</li>

</ul>




<ol start="3">

 <li>Compare the recognition accuracy rate of the test set, total number of parameters, and total number of multiplications of this CNN to those in Lab 7 (neural network). Analyze your findings and explain why you obtain different (or similar) results.</li>

</ol>

<strong>Demo/Explain to TA : </strong>

<ol>

 <li>How do you build the layers of the convolutional neural network?</li>

 <li>How do you do prediction for the test samples?</li>

 <li>How do you calculate the number of parameters and the number of multiplications?</li>

</ol>


