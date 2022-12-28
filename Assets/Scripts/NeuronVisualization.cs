using System.Linq;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetworkExample
{
    [RequireComponent(typeof(NeuralNetwork))]
    [RequireComponent(typeof(TrainingNN))]
    public class NeuronVisualization : MonoBehaviour
    {
        NeuralNetwork neuralNetwork;
        TrainingNN training;
        public GameObject prefab;
        public float scale = 1;
        public float spacing = 1;
        private List<GameObject> inputNeurons;
        private List<GameObject> hiddenNeurons;
        private List<GameObject> outputNeurons;
        private void Awake()
        {
            neuralNetwork = GetComponent<NeuralNetwork>();
            training = GetComponent<TrainingNN>();
        }
        private void Start()
        {
            // Create a sphere for each neuron
            GameObject _gameObject = new GameObject();
            var inputLayer = Instantiate(_gameObject, transform);
            inputLayer.name = "Input Layer";
            inputLayer.transform.localPosition = Vector3.left * scale;
            CreateNeurons(ref inputNeurons, neuralNetwork.InputLayerSize, inputLayer.transform);

            var hiddenLayer = Instantiate(_gameObject, transform);
            hiddenLayer.name = "Hidden Layer";
            hiddenLayer.transform.localPosition = Vector3.zero * scale;
            CreateNeurons(ref hiddenNeurons, neuralNetwork.HiddenLayerSize, hiddenLayer.transform);

            var outputLayer = Instantiate(_gameObject, transform);
            outputLayer.name = "Output Layer";
            outputLayer.transform.localPosition = Vector3.right * scale;
            CreateNeurons(ref outputNeurons, neuralNetwork.OutputLayerSize, outputLayer.transform);
            Destroy(_gameObject);
        }
        private void CreateNeurons(ref List<GameObject> list, int size, Transform parent)
        {
            list = new List<GameObject>();
            for (int i = 0; i < size; i++)
            {
                var neuron = GameObject.Instantiate(prefab, parent);

                neuron.transform.localPosition = Vector3.up * i * spacing;
                neuron.transform.localScale = Vector3.one * scale;
                list.Add(neuron);
            }
        }
        private void Update()
        {
            if (!training.IsTraining) return;
            // Set the values of the neurons based on the activations of the neural network
            SetNeuronValues(inputNeurons, neuralNetwork.Input);
            SetNeuronValues(hiddenNeurons, neuralNetwork.ActivationsHidden);
            SetNeuronValues(outputNeurons, neuralNetwork.ActivationsOutput);
        }
        private void SetNeuronValues(List<GameObject> neurons, List<double> values)
        {
            if (values.Count == 0) return;
            // Normalize the values by dividing them by the maximum value in the list
            var normalizedValues = values.Select(v => v / values.Max()).ToList();

            for (int i = 0; i < neurons.Count; i++)
            {
                neurons[i].GetComponent<Renderer>().material.color = Color.Lerp(Color.red, Color.blue, (float)values[i]);
            }
        }
    }

}
