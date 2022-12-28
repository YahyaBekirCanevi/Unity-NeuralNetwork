using System.Collections.Generic;
using System;
using System.Linq;
using UnityEngine;

namespace NeuralNetworkExample
{
    public class NeuralNetwork : MonoBehaviour
    {
        #region Variables
        // Number of input, hidden, output layers and learning rate
        int inputLayerSize;
        int hiddenLayerSize;
        int outputLayerSize;
        double learningRate;

        // Weights and biases
        List<List<double>> weightsInputToHidden;
        List<double> biasesHidden;
        List<List<double>> weightsHiddenToOutput;
        List<double> biasesOutput;

        // Cost Values
        [SerializeField] List<double> input;
        [SerializeField] List<double> activationsHidden;
        [SerializeField] List<double> activationsOutput;
        // Activation function
        Func<double, double> activationFunction;
        Func<double, double> activationFunctionDerivative;
        #endregion
        #region Properties
        public int InputLayerSize { get => inputLayerSize; set => inputLayerSize = value; }
        public int HiddenLayerSize { get => hiddenLayerSize; set => hiddenLayerSize = value; }
        public int OutputLayerSize { get => outputLayerSize; set => outputLayerSize = value; }
        public double LearningRate { get => learningRate; set => learningRate = value; }
        public List<List<double>> WeightsInputToHidden { get => weightsInputToHidden; }
        public List<List<double>> WeightsHiddenToOutput { get => weightsHiddenToOutput; }
        public List<double> Input { get => input; set => input = value; }
        public List<double> ActivationsHidden { get => activationsHidden; set => activationsHidden = value; }
        public List<double> ActivationsOutput { get => activationsOutput; set => activationsOutput = value; }
        public Func<double, double> ActivationFunction { get => activationFunction; set => activationFunction = value; }
        public Func<double, double> ActivationFunctionDerivative { get => activationFunctionDerivative; set => activationFunctionDerivative = value; }
        #endregion
        #region Methods
        public void Initialize(
            int inputLayerSize,
            int hiddenLayerSize,
            int outputLayerSize,
            double learningRate,
            Func<double, double> activationFunction,
            Func<double, double> activationFunctionDerivative
        )
        {
            this.inputLayerSize = inputLayerSize;
            this.hiddenLayerSize = hiddenLayerSize;
            this.outputLayerSize = outputLayerSize;
            this.learningRate = learningRate;
            this.activationFunction = activationFunction;
            this.activationFunctionDerivative = activationFunctionDerivative;

            // Initialize weights and biases with random values
            weightsInputToHidden = Enumerable.Range(0, hiddenLayerSize).Select(_ => Enumerable.Range(0, inputLayerSize).Select(__ => (double)UnityEngine.Random.Range(-1f, 1f)).ToList()).ToList();
            biasesHidden = Enumerable.Range(0, hiddenLayerSize).Select(_ => (double)UnityEngine.Random.Range(-1f, 1f)).ToList();

            weightsHiddenToOutput = Enumerable.Range(0, outputLayerSize).Select(_ => Enumerable.Range(0, hiddenLayerSize).Select(__ => (double)UnityEngine.Random.Range(-1f, 1f)).ToList()).ToList();
            biasesOutput = Enumerable.Range(0, outputLayerSize).Select(_ => (double)UnityEngine.Random.Range(-1f, 1f)).ToList();
        }
        public List<double> Predict(List<double> newInput)
        {
            input = newInput;
            // Calculate the activations of the hidden layer
            activationsHidden = Enumerable.Range(0, hiddenLayerSize)
                .Select(i => weightsInputToHidden[i]
                    .Zip(input, (w, x) => w * x)
                    .Sum() + biasesHidden[i])
                .Select(activationFunction)
                .ToList();

            // Calculate the activations of the output layer
            activationsOutput = Enumerable.Range(0, outputLayerSize)
                .Select(i => weightsHiddenToOutput[i]
                    .Zip(activationsHidden, (w, x) => w * x)
                    .Sum() + biasesOutput[i])
                .Select(activationFunction)
                .ToList();

            return activationsOutput;
        }
        public void Backpropagate(List<double> input, List<double> target)
        {
            // Calculate the activations and errors for the hidden and output layers
            var activationsHidden = Enumerable.Range(0, hiddenLayerSize)
                .Select(i => weightsInputToHidden[i]
                    .Zip(input, (w, x) => w * x)
                    .Sum() + biasesHidden[i])
                .Select(activationFunction)
                .ToList();

            var errorsHidden = Enumerable.Range(0, hiddenLayerSize).Select(_ => 0.0).ToList();

            var activationsOutput = Enumerable.Range(0, outputLayerSize)
                .Select(i => weightsHiddenToOutput[i]
                    .Zip(activationsHidden, (w, x) => w * x)
                    .Sum() + biasesOutput[i])
                .Select(activationFunction)
                .ToList();

            var errorsOutput = Enumerable.Range(0, outputLayerSize)
                .Select(i => activationsOutput[i] - target[i])
                .ToList();

            // Calculate the error gradients for the output and hidden layers
            var errorGradientsOutput = Enumerable.Range(0, outputLayerSize)
                .Select(i => errorsOutput[i] * activationFunctionDerivative(activationsOutput[i]))
                .ToList();

            var errorGradientsHidden = Enumerable.Range(0, hiddenLayerSize)
                .Select(i => Enumerable.Range(0, outputLayerSize)
                    .Select(j => errorGradientsOutput[j] * weightsHiddenToOutput[j][i])
                    .Sum() * activationFunctionDerivative(activationsHidden[i]))
                .ToList();

            // Update the weights and biases
            weightsInputToHidden = weightsInputToHidden
                .Zip(errorGradientsHidden, (w, g) => w
                    .Zip(input, (v, x) => v - learningRate * g * x)
                    .ToList())
                .ToList();
            biasesHidden = biasesHidden
                .Zip(errorGradientsHidden, (b, g) => b - learningRate * g)
                .ToList();

            weightsHiddenToOutput = weightsHiddenToOutput
                .Zip(errorGradientsOutput, (w, g) => w
                    .Zip(activationsHidden, (v, x) => v - learningRate * g * x)
                    .ToList())
                .ToList();
            biasesOutput = biasesOutput
                .Zip(errorGradientsOutput, (b, g) => b - learningRate * g)
                .ToList();
        }
        #endregion
    }
}
