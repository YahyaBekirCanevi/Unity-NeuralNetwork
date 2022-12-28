using System;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetworkExample
{
    [RequireComponent(typeof(NeuralNetwork))]
    public class TrainingNN : MonoBehaviour
    {
        [SerializeField] int inputLayerSize;
        [SerializeField] int hiddenLayerSize;
        [SerializeField] int outputLayerSize;
        [SerializeField] double learningRate;
        [SerializeField] double decayRate;
        // Neural network instance
        NeuralNetwork neuralNetwork;
        // Training data
        List<(List<double>, List<double>)> trainingData;

        // Variables
        [SerializeField] bool isBackPropagate = true;
        [Tooltip("S Key starts")] public bool IsTraining = false;

        private void Awake()
        {
            neuralNetwork = GetComponent<NeuralNetwork>();
            // Sigmoid activation function
            Func<double, double> sigmoid = x => 1 / (1 + Math.Exp(-x));
            // Sigmoid derivative function
            Func<double, double> sigmoidDerivative = y => y * (1 - y);

            // Initialize neural network and training data
            neuralNetwork.Initialize(
                inputLayerSize: inputLayerSize,
                hiddenLayerSize: hiddenLayerSize,
                outputLayerSize: outputLayerSize,
                learningRate: learningRate,
                activationFunction: sigmoid,
                activationFunctionDerivative: sigmoidDerivative
            );
            trainingData = new List<(List<double>, List<double>)>
            {
                //(input, output)
                (new List<double> { 0.1, 0.2 },new List<double> { 0.9, 0.1 }),
                (new List<double> { 0.3, 0.4 },new List<double> { 0.8, 0.2 }),
                (new List<double> { 0.5, 0.6 },new List<double> { 0.7, 0.3 }),
                (new List<double> { 0.7, 0.8 },new List<double> { 0.6, 0.4 }),
                (new List<double> { 0.9, 1.0 },new List<double> { 0.5, 0.5 })
            };
        }
        void Update()
        {
            if (Input.GetKeyDown(KeyCode.S)) IsTraining = !IsTraining;
            if (!IsTraining) return;

            // Loop through the training data and perform backpropagation
            foreach (var (input, target) in trainingData)
            {
                // Make a prediction using the input data
                neuralNetwork.Predict(input);

                if (isBackPropagate)
                {
                    // Use sigmoid derivative in the backpropagation function
                    neuralNetwork.Backpropagate(input, target);
                }
            }
            neuralNetwork.LearningRate = this.learningRate = this.learningRate * (1 - decayRate);
        }
    }

}