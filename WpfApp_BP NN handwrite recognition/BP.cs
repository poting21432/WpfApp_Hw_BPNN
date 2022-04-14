using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using WpfApp_BP_NN_handwrite_recognition;

namespace NN_MNIST
{
    public class BP
    {
        public bool StopSig = false;
        public bool Training { get; private set; } = false;
        public bool Testing { get; private set; } = false;
        public int TrainSize;
        public int TestSize;

        public double LearningRate;
        public double[,] Weights_Input;
        public double[,,] Weights_Hidden;
        public double[,] Weights_Output;

        public double[] InputNodes;
        public double[,] HiddenNodes;
        public double[] OutputNodes;

        public double[,] Delta_Input;
        public double[,,] Delta_Hidden;
        public double[,] Delta_Output;

        public double[] Derivative_Input;
        public double[,] Derivative_Hidden;
        public double[] Derivative_Output;

        public int HiddenSize = 60;
        public int HiddenLayersSize = 2;
        public int InputLayersSize = 28 * 28;//14 * 14;// 
        public int OutputLayersSize = 10;
        public const double LearningScale = 1;
        public int Iterative = 100;
        public int LogCounter = 1000;
        public const double LossThreshold = 0.001;

        public BP()
        {
            Weights_Input = new double[InputLayersSize, HiddenSize];
            Weights_Hidden = new double[HiddenLayersSize - 1, HiddenSize, HiddenSize];
            Weights_Output = new double[HiddenSize, OutputLayersSize];

            Delta_Input = new double[InputLayersSize, HiddenSize];
            Delta_Hidden = new double[HiddenLayersSize - 1, HiddenSize, HiddenSize];
            Delta_Output = new double[HiddenSize, OutputLayersSize];

            InputNodes = new double[InputLayersSize];
            HiddenNodes = new double[HiddenLayersSize, HiddenSize];
            OutputNodes = new double[OutputLayersSize];

            Derivative_Input = new double[OutputLayersSize];
            Derivative_Hidden = new double[HiddenLayersSize, HiddenSize];
            Derivative_Output = new double[OutputLayersSize];

            Random rand = new Random();
            Weights_Input.ForEach((j, k) => Weights_Input[j, k] = rand.NextDouble() * 2.0 - 1.0);
            Weights_Hidden.ForEach((i, j, k) => Weights_Hidden[i, j, k] = rand.NextDouble() * 2.0 - 1.0);
            Weights_Output.ForEach((j, k) => Weights_Output[j, k] = rand.NextDouble() * 2.0 - 1.0);
            HiddenNodes.ForEach((j, k) => HiddenNodes[j, k] = 0);
        }
        private static Random rng = new Random();

        private void Shuffle<T>(List<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
        /// <summary>
        /// Return Cost
        /// </summary>
        public void Train(List<Image> ListTrain)
        {
            if (Training || Testing)
            {
                Global.AppendLog(string.Format("Is Training or Testing!\r"));
                return;
            }
            new Thread(() =>
            {
                Training = true;
                DateTime dtStart = DateTime.Now;
                for (int i = 0; i < Iterative; i++)
                {
                    if (StopSig)
                        break;
                    Shuffle(ListTrain);
                    double T = ListTrain.Count;
                    double TotalLoss = 0; int Passed = 0; double lrate = 0;
                    for (int t =1;t<= T;t++)
                    {
                        if (StopSig)
                            break;
                        Image img = ListTrain[t-1];

                        double loss = Forward(img);
                        LearningRate = LearningScale * loss;
                        if (img.Label != FindMax())
                            LearningRate *= 1.2;
                        else
                            LearningRate *= 0.8;

                        if (loss <= LossThreshold && img.Label == FindMax())
                            Passed++;
                        else
                        {
                            ErrorBP(img);
                            TotalLoss += loss;
                            lrate += LearningRate;
                            UpdateWeight();
                        }
                        if (t % LogCounter == 0)
                        {
                            TotalLoss /= (double)(LogCounter - Passed);
                            lrate /= (double)(LogCounter - Passed);
                            Global.AppendLog(string.Format("iter:{0}, t:{1}, LRate:{2}, loss:{3}, Passed={4}, time_elap={5}s\r",
                                                            i, t, LearningRate.ToString("F2"), TotalLoss.ToString("F4"), Passed,
                                                            (DateTime.Now - dtStart).TotalSeconds.ToString("F2")));
                            TotalLoss = 0;
                            lrate = 0;
                            Passed = 0;

                        }
                    }
                    int n = 1; int AC = 0; int WA = 0;
                    Global.ImagesTrain.ForEach((img) =>
                    {
                        Forward(img);
                        int predict = FindMax();
                        if (predict == img.Label)
                            AC++;
                        else
                            WA++;
                        n++;
                    });
                    Global.AppendLog(string.Format("TestSet AC = {0}, WA = {1}, AC_Rate={2}\r",
                                        AC, WA, ((double)AC / (AC + WA)).ToString("F3")));
                }
                Training = false; StopSig = false;
            }){ IsBackground = true, Priority = ThreadPriority.Highest }.Start();
            
        }

        private void UpdateWeight()
        {
            Weights_Input.ForEach((j, k) => Weights_Input[j, k] += Delta_Input[j, k]);
            Weights_Hidden.ForEach((i, j, k) => Weights_Hidden[i, j, k] += Delta_Hidden[i, j, k]);
            Weights_Output.ForEach((j, k) => Weights_Output[j, k] += Delta_Output[j, k]);
        }
        public int TestOne(List<Image> ListTest, int id)
        {
            Forward(ListTest[id]);
            return FindMax();
        }
        public void Test(List<Image> ListTest)
        {
            if (Training || Testing)
            {
                Global.AppendLog(string.Format("Is Training or Testing!\r"));
                return;
            }
            new Thread(() =>
            {
                Testing = true;
                int n = 1; int AC = 0; int WA = 0;
                ListTest.ForEach((img) =>
                {
                    Forward(img);
                    img.Predict = FindMax();
                    if (img.Predict == img.Label)
                        AC++;
                    else
                        WA++;
                    n++;

                    
                });
                Global.AppendLog(string.Format("TestSet AC = {0}, WA = {1}, AC_Rate={2}\r",
                                    AC, WA, ((double)AC / (AC + WA)).ToString("F3")));

                Testing = false;
            }) { IsBackground = true, Priority = ThreadPriority.AboveNormal }.Start();
        }
        /* //https://docs.microsoft.com/zh-tw/dotnet/standard/parallel-programming/how-to-speed-up-small-loop-bodies
        private void PForward(Image input)
        {
            Parallel.For(0, input.Data.GetLength(0), (i) => {
                Parallel.For(0, input.Data.GetLength(1), (j) => {
                    InputNodes[i * input.Data.GetLength(1) + j] = NN_Helper.Formalize(input.Data[i, j]);
                });
            });
            Parallel.For(0, Weights_Input.GetLength(1), (j) => {
                double psigma = 0;
                Parallel.For(0, Weights_Input.GetLength(0), (i) => {
                    psigma += Weights_Input[i, j] * InputNodes[i];
                });
                HiddenNodes[0, j] = NN_Helper.Sigmoid(psigma);
            });
            Parallel.For(0, Weights_Hidden.GetLength(0), (i) => {
                Parallel.For(0, Weights_Hidden.GetLength(2), (k) => {
                    double psigma = 0;
                    Parallel.For(0, Weights_Hidden.GetLength(1), (j) => {
                        psigma += (Weights_Hidden[i, j, k] * HiddenNodes[i, j]);
                    });
                    HiddenNodes[i + 1, k] = NN_Helper.Sigmoid(psigma);
                });
            });
        }*/
        private double Forward(Image input)
        {
            double sigma = 0;
            /*/ i = 14  j = 14 -> Set input to 0.0~1.0
            InputNodes.ForEach((i) => {
                i *= 2;
                int total = input.Data[i / 28, i % 28] + input.Data[i / 28, i % 28 + 1] +
                            input.Data[i / 28 + 1, i % 28] + input.Data[i / 28 + 1, i % 28 + 1];
                
                //double new_shift = (total * total) / (255 * 255);
                //total = (int)(total * new_shift);
                i /= 2;
                InputNodes[i] = NN_Helper.Formalize((byte)(total / 4));
            }
            );//*/
            // i = 28  j = 28 -> Set input to 0.0~1.0
            input.Data.ForEach((i, j) => InputNodes[i * input.Data.GetLength(1) + j] = NN_Helper.Formalize(input.Data[i, j]));
            /// i =InputLayersSize j = HiddenSize  -> First Layer to Hidden Layer

            
            for (int j = 0; j < Weights_Input.GetLength(1); j++)
            {
                sigma = 0;
                for (int i = 0; i < Weights_Input.GetLength(0); i++)
                    sigma += Weights_Input[i, j] * InputNodes[i];
                HiddenNodes[0, j] = NN_Helper.Sigmoid(sigma);
            }
            /// i = HiddenLayersSize - 1 j = HiddenSize k = HiddenSize  -> After Second Layer to 0.0~1.0

            for (int i = 0; i < Weights_Hidden.GetLength(0); i++)
                for (int k = 0; k < Weights_Hidden.GetLength(2); k++)
                {
                    sigma = 0;
                    for (int j = 0; j < Weights_Hidden.GetLength(1); j++)
                         sigma += (Weights_Hidden[i, j, k] * HiddenNodes[i, j]);
                    HiddenNodes[i + 1, k] = NN_Helper.Sigmoid(sigma);
                }
            for (int j = 0; j < Weights_Output.GetLength(1); j++)
            {
                sigma = 0;
                for (int i = 0; i < Weights_Output.GetLength(0); i++)
                    sigma += Weights_Output[i, j] * HiddenNodes[HiddenLayersSize - 1, i];
                OutputNodes[j] = NN_Helper.Sigmoid(sigma);
            }
            double[] Des = Desired(input.Label);

            double loss = 0;
            OutputNodes.ForEach((i) =>
            {
                double diff = OutputNodes[i] - Des[i];
                loss += (diff * diff);
            });
            loss *= 0.5;
            return loss;
        }

        private void ErrorBP(Image input)
        {
            //https://www.andreaperlato.com/theorypost/the-learning-rate/
            double[] Des = Desired(input.Label);
            ///Output Layer
            /// Count last layer
            Derivative_Output.ForEach((i) => Derivative_Output[i] = NN_Helper.Derivative(OutputNodes[i]) * (Des[i] - OutputNodes[i]));
            Delta_Output.ForEach((i,j) => Delta_Output[i,j] = 
                LearningRate * Derivative_Output[j] * HiddenNodes[HiddenLayersSize -1, i]);
            
            /// Hidden layer
            ///First Derivative of last Hidden layer
            int last_b = HiddenLayersSize - 1;
            for (int i = 0; i < Weights_Output.GetLength(0); i++)
            {
                double sigma = 0;
                int size = Derivative_Output.GetLength(0);
                Derivative_Output.ForEach((j) =>
                    sigma += Derivative_Output[j] * Weights_Output[i, j]);
                Derivative_Hidden[last_b, i] = NN_Helper.Derivative(HiddenNodes[last_b, i]) * sigma;
            }

            ///Other hidden Layers
            for (int l = 0; l < Weights_Hidden.GetLength(0); l++)
            {
                int b = Weights_Hidden.GetLength(0) - l - 1;
                for (int i = 0; i < Weights_Hidden.GetLength(1); i++)
                {
                    double sigma = 0;
                    for (int j = 0; j < Derivative_Hidden.GetLength(1); j++)
                        sigma += Derivative_Hidden[b + 1, j] * Weights_Hidden[b, i, j];
                    Derivative_Hidden[b, i] = NN_Helper.Derivative(HiddenNodes[b, i]) * sigma;
                }
                
                for (int i = 0; i < Delta_Hidden.GetLength(1); i++)
                   for (int j = 0; j < Delta_Hidden.GetLength(2); j++)
                        Delta_Hidden[b, i, j] = LearningRate * Derivative_Hidden[b + 1, j] * HiddenNodes[b, i];
            }
            ///Input layer
            Delta_Input.ForEach((i, j) => Delta_Input[i, j] =
                LearningRate * Derivative_Hidden[0, j] * InputNodes[i]);
        }

        public double[] Desired(int Label)
        {
            double[] des= new double[OutputLayersSize];
            des.ForEach((i) => des[i] = 0.0);
            des[Label] = 1.0;
            return des;
        }
        public int FindMax()
        {
            List<double> output = OutputNodes.ToList();
            return output.IndexOf(output.Max());
        }
    }
    public static class NN_Helper
    {
        public static double Sigmoid(double input)
        {
            return 1 / (1 + (Math.Exp(-input)));
        }

        public static double Derivative(double fx)
        {
            return fx * (1 - fx);
        }

        public static double Formalize(byte input)
        {
            return (input / 255.0) * 1.0;
        }
    }
}
