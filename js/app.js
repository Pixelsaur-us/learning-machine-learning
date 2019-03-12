// Create a sequential model
const model = tf.sequential();

// Add a hidden layer 
model.add(tf.layers.dense({
  units: 4,
  inputShape: [3],
  activation:'sigmoid',
}));

// Add output layer 
model.add(tf.layers.dense({
  units: 2,
  activation: 'softmax',
}));

// Configure the model for training
model.compile({
  optimizer: tf.train.sgd(0.5),
  loss: tf.losses.softmaxCrossEntropy,
});

// Test data
const xs = tf.tensor2d([
  [0.0, 0.0, 0.0],
  [0.5, 0.5, 0.5],
  [1.0, 1.0, 1.0],
]);

// Test results
const ys = tf.tensor2d([
  [0.0, 1.0],
  [0.5, 0.5],
  [1.0, 0.0],
]);

async function train(iterations) {
  for (let i = 0; i < iterations; ++i) {
    await model.fit(xs, ys, {
      shuffle: true,
      epochs: 30,
    }).then(() => {
      model.predict(xs).print();
    });
  }
};
train(50);