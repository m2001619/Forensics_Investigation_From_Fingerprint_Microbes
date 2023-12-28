// Import required libraries
const fs = require('fs');
const Papa = require('papaparse');
const KNN = require('ml-knn');

// Function to load CSV data and filter the features and labels
function loadData(filename) {
    const csvData = fs.readFileSync(filename, 'utf-8');
    const parsedData = Papa.parse(csvData, {header: true});
    const dataset = parsedData.data.map(i => Object.values(i));
    const labels = dataset[0];
    const features = transposeArray(dataset.slice(1));
    return [features, labels];
}

// Load dataset then fetch features and labels
const [features, labels] = loadData(`./otu.csv`);


// Initialize KNN classifier and train the model
const k = 3; // Adjust k value as needed
const knn = new KNN(features, labels, {k});

// Split the dataset into training and testing sets ,
const assumeTraining = 0.7; // 70% training and 30% testing data
const splitIndex = Math.floor(assumeTraining * features.length);
const X_test = features.slice(splitIndex);
const y_test = labels.slice(splitIndex);
const X_train = features.slice(0, splitIndex);
const y_train = labels.slice(0, splitIndex);

// Make predictions on the train set
const trainPredictions = X_train.map(dataPoint => knn.predict(dataPoint));
const testPredictions = X_test.map(dataPoint => knn.predict(dataPoint));

// Print the output for train data
console.log(`For K = ${k} and and Assume Training = ${assumeTraining}\n`);

const trainAccuracy = calculateAccuracy(trainPredictions, y_train);
console.log('Train Data Accuracy:', trainAccuracy);
const trainSensitivity = calculateSensitivity(trainPredictions, y_train);
console.log('Train Data Sensitivity:', trainSensitivity);
const trainSpecificity = calculateSpecificity(trainPredictions, y_train);
console.log('Train Data Specificity:', trainSpecificity);

console.log("*".repeat(50));

// Print the output for test data
const testAccuracy = calculateAccuracy(testPredictions, y_test);
console.log('Test Data Accuracy:', testAccuracy);
const testSensitivity = calculateSensitivity(testPredictions, y_test);
console.log('Test Data Sensitivity:', testSensitivity);
const testSpecificity = calculateSpecificity(testPredictions, y_test);
console.log('Test Data Specificity:', testSpecificity);

/** Start Handler Functions **/
function calculateAccuracy(predictions, actualLabels) {
    const correct = predictions.filter((prediction, idx) => prediction === actualLabels[idx]);
    return correct.length / actualLabels.length;
}

function calculateSensitivity(predictions, actualLabels) {
    const truePositives = predictions.reduce((count, prediction, idx) => {
        if (prediction === 'left' && actualLabels[idx] === 'left') {
            return count + 1;
        }
        return count;
    }, 0);

    const totalLeft = actualLabels.filter(label => label === 'left').length;

    return truePositives / totalLeft;
}

function calculateSpecificity(predictions, actualLabels) {
    const trueNegatives = predictions.reduce((count, prediction, idx) => {
        if (prediction === 'right' && actualLabels[idx] === 'right') {
            return count + 1;
        }
        return count;
    }, 0);

    const totalRight = actualLabels.filter(label => label === 'right').length;

    return trueNegatives / totalRight;
}

// Function accept array and  return the transpose for given array
function transposeArray(arr) {
    return arr[0].map((_, colIndex) => arr.map(row => +row[colIndex]));
}

/** End Handler Functions **/
