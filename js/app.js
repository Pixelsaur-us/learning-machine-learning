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
  optimizer: tf.train.sgd(0.15),
  loss: tf.losses.softmaxCrossEntropy,
});

async function train(iterations) {
  const inputs = trainingData.map((data) => {
    return [data.r/255, data.g/255, data.b/255];
  });
  const results = trainingData.map((data) => {
    return data.selected;
  });
  const x = tf.tensor(inputs, [inputs.length, 3], 'float32');
  const y = tf.tensor(results, [results.length, 2], 'float32');
  for (let i = 0; i < iterations; ++i) {
    const response = await model.fit(x, y, {
      shuffle: true,
      epochs: 30,
    });
    console.log(response.history.loss);
  }
};

$('.slider').slider({showInstruction: false});

const pickColour = async (r, g, b, bw) => {
  const x = tf.tensor2d([r/255,g/255,b/255], [1,3]);
  const y = tf.tensor2d(bw, [1,2]);
  await model.fit(x, y, {epochs: 50}).then((response) => {
    console.log(response.history.loss[0]);
    changeBackgroundColour();
  });
}

const predictColour = async (r, g, b) => {
  let input = tf.tensor2d([r/255,g/255,b/255], [1,3])
  let output =  await model.predict(input).data();
  console.log('output: ', output);
  if (output[0] > output[1]) {
    $('#black').css('border', 'solid');
    $('#white').css('border', 'none');
  } else {
    $('#white').css('border', 'solid');
    $('#black').css('border', 'none');
  }
}

function changeBackgroundColour() {
  // Generate new random colour
  const r = Math.floor(Math.random()*255);
  const g = Math.floor(Math.random()*255);
  const b = Math.floor(Math.random()*255);
  
  // Prediction based on new colour
  predictColour(r, g, b);

  $('body').css({
    'background-color' : `rgb(${r}, ${g}, ${b})`,
  });
}

function getBackgroundColour() {
  const currentColour = $('body').css('background-color');
  const rgb = currentColour.replace(/\s/g,'').match(/^rgba?\((\d+),(\d+),(\d+)/i);
  return {r: rgb[1], g: rgb[2], b: rgb[3]}
}

let trainingData = [];

$('.selection .button').on('click', (e) => {
  // Reset the slider to the middle
  $('.slider').slider();
  
  // Get current background colour
  const rgb = getBackgroundColour();
  const selected = e.currentTarget.id === 'black' ? [1,0] : [0,1];

  // Save training data
  trainingData.push({...rgb, selected});

  // Fit the selection to the model
  pickColour(rgb.r, rgb.g, rgb.b, selected);
})

$('document').ready(() => {
  // changeBackgroundColour();
})