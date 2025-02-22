using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;

class Program
{
    // Двовимірні точки
    private static List<(double X, double Y, int Label)> points = new List<(double, double, int)>();
    private static double[] weights;
    private static double bias;

    // Багатовимірні точки
    private static List<(double[] Features, int Label)> multidimensionalPoints = new List<(double[], int)>();
    private static double[] weightsMulti;
    private static double biasMulti;

    static void Main()
    {
        // Двовимірний випадок
        GeneratePoints();
        LabelPoints();
        DrawPoints();
        CalculateLinearRegression();
        TrainPerceptron();
        GetWeights();
        EvaluatePerceptron();
        TestActivationFunctions();

        // Багатовимірний випадок
        GenerateMultidimensionalPoints();
        TrainMultidimensionalPerceptron();
        GetMultidimensionalWeights();
        EvaluateMultidimensionalPerceptron();
    }

    // --- Двовимірний випадок ---
    static void GeneratePoints()
    {
        Random random = new Random();
        for (int i = 0; i < 100; i++)
            points.Add((random.NextDouble() * 0.5, random.NextDouble() * 0.5, 0));
        for (int i = 0; i < 100; i++)
            points.Add((0.5 + random.NextDouble() * 0.5, 0.5 + random.NextDouble() * 0.5, 1));
    }

    static void LabelPoints()
    {
        Random random = new Random();
        for (int i = 0; i < 100; i++)
            points.Add((random.NextDouble() * 0.5, random.NextDouble() * 0.5, 0)); // Мітка 0
        for (int i = 0; i < 100; i++)
            points.Add((0.5 + random.NextDouble() * 0.5, 0.5 + random.NextDouble() * 0.5, 1)); // Мітка 1
        Console.WriteLine("Точки позначено.");
    }

    static void DrawPoints()
    {
        int width = 500, height = 500;
        using (Bitmap bitmap = new Bitmap(width, height))
        using (Graphics graphics = Graphics.FromImage(bitmap))
        {
            graphics.Clear(Color.White);

            foreach (var point in points)
            {
                Color color = point.Label == 0 ? Color.Red : Color.Blue;
                int x = (int)(point.X * width);
                int y = (int)((1 - point.Y) * height);
                graphics.FillEllipse(new SolidBrush(color), x, y, 5, 5);
            }

            string outputPath = Path.Combine(Environment.CurrentDirectory, "points.png");
            bitmap.Save(outputPath, System.Drawing.Imaging.ImageFormat.Png);
        }
    }

    static void CalculateLinearRegression()
    {
        double meanX = points.Average(p => p.X);
        double meanY = points.Average(p => p.Y);

        double numerator = points.Sum(p => (p.X - meanX) * (p.Y - meanY));
        double denominator = points.Sum(p => Math.Pow(p.X - meanX, 2));

        double slope = numerator / denominator;
        double intercept = meanY - slope * meanX;

        Console.WriteLine($"Рiвняння лiнiйної регресiї: y = {slope}x + {intercept}");
    }

    static void TrainPerceptron(int epochs = 100, double learningRate = 0.1)
    {
        Random random = new Random();
        weights = new double[2] { random.NextDouble(), random.NextDouble() };
        bias = random.NextDouble();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var point in points)
            {
                double output = weights[0] * point.X + weights[1] * point.Y + bias;
                int predicted = output >= 0 ? 1 : 0;
                int error = point.Label - predicted;

                weights[0] += learningRate * error * point.X;
                weights[1] += learningRate * error * point.Y;
                bias += learningRate * error;
            }
        }
        Console.WriteLine("Персептрон навчено.");
    }

    static void GetWeights()
    {
        Console.WriteLine($"Ваги: w1 = {weights[0]}, w2 = {weights[1]}, Зміщення = {bias}");
    }

    static void EvaluatePerceptron()
    {
        int correct = 0;
        foreach (var point in points)
        {
            double output = weights[0] * point.X + weights[1] * point.Y + bias;
            int predicted = output >= 0 ? 1 : 0;
            if (predicted == point.Label)
                correct++;
        }
        double accuracy = (double)correct / points.Count * 100;
        Console.WriteLine($"Точнiсть персептрона (двовимiрний випадок): {accuracy}%");
    }

    static void TestActivationFunctions()
    {
        Console.WriteLine("Тестування функцiй активацiї:");
        double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        double Tanh(double x) => Math.Tanh(x);
        double ReLU(double x) => Math.Max(0, x);

        foreach (var point in points)
        {
            double input = weights[0] * point.X + weights[1] * point.Y + bias;
            Console.WriteLine($"Точка ({point.X}, {point.Y}) -> Sigmoid: {Sigmoid(input)}, Tanh: {Tanh(input)}, ReLU: {ReLU(input)}");
        }
    }

    // --- Багатовимірний випадок ---

    static void GenerateMultidimensionalPoints()
    {
        Console.WriteLine("Генерацiя багатовимiрних точок iз Гауссiвським розподiлом");
        Random random = new Random();
        int dimensions = 3;
        int pointsPerClass = 100;

        for (int i = 0; i < pointsPerClass; i++)
        {
            double[] features = new double[dimensions];
            for (int j = 0; j < dimensions; j++)
                features[j] = random.NextDouble();
            multidimensionalPoints.Add((features, 0));
        }

        for (int i = 0; i < pointsPerClass; i++)
        {
            double[] features = new double[dimensions];
            for (int j = 0; j < dimensions; j++)
                features[j] = 0.5 + random.NextDouble() * 0.5;
            multidimensionalPoints.Add((features, 1));
        }
    }

    static void TrainMultidimensionalPerceptron(int epochs = 100, double learningRate = 0.1)
    {
        Random random = new Random();
        int dimensions = multidimensionalPoints[0].Features.Length;
        weightsMulti = new double[dimensions];
        for (int i = 0; i < dimensions; i++)
            weightsMulti[i] = random.NextDouble();

        biasMulti = random.NextDouble();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var point in multidimensionalPoints)
            {
                double output = 0;
                for (int i = 0; i < dimensions; i++)
                    output += weightsMulti[i] * point.Features[i];

                output += biasMulti;

                int predicted = output >= 0 ? 1 : 0;
                int error = point.Label - predicted;

                for (int i = 0; i < dimensions; i++)
                    weightsMulti[i] += learningRate * error * point.Features[i];

                biasMulti += learningRate * error;
            }
        }
        Console.WriteLine("Персептрон навчено.");
    }

    static void GetMultidimensionalWeights()
    {
        Console.WriteLine("Ваги персептрона (багатовимiрний випадок):");
        for (int i = 0; i < weightsMulti.Length; i++)
            Console.WriteLine($"w{i + 1} = {weightsMulti[i]}");
        Console.WriteLine($"Зсув = {biasMulti}");
    }

    static void EvaluateMultidimensionalPerceptron()
    {
        int correct = 0;
        foreach (var point in multidimensionalPoints)
        {
            double output = 0;
            for (int i = 0; i < point.Features.Length; i++)
                output += weightsMulti[i] * point.Features[i];

            output += biasMulti;

            int predicted = output >= 0 ? 1 : 0;
            if (predicted == point.Label)
                correct++;
        }
        double accuracy = (double)correct / multidimensionalPoints.Count * 100;
        Console.WriteLine($"Точнiсть персептрона (багатовимiрний випадок): {accuracy}%");
    }
}
