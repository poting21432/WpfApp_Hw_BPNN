using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_MNIST
{
    //https://stackoverflow.com/questions/49407772/reading-mnist-database
    public class Image
    {
        public byte Label { get; set; }
        public byte[,] Data { get; set; }

        public int Predict { get; set; } = -1;
    }
    public static class MnistReader
    {
        private const string TrainImages = "./train-images.idx3-ubyte";
        private const string TrainLabels = "./train-labels.idx1-ubyte";
        private const string TestImages = "./t10k-images.idx3-ubyte";
        private const string TestLabels = "./t10k-labels.idx1-ubyte";

        public static IEnumerable<Image> ReadTrainingData()
        {
            foreach (var item in Read(TrainImages, TrainLabels))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData()
        {
            foreach (var item in Read(TestImages, TestLabels))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var arr = new byte[height, width];
                arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                yield return new Image()
                {
                    Data = arr,
                    Label = labels.ReadByte()
                };
            }
        }

        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
                for (int h = 0; h < source.GetLength(1); h++)
                    action(w, h);
        }

        public static void ForEach<T>(this T[,,] source, Action<int,int, int> action)
        {
            for (int l = 0; l < source.GetLength(0); l++)
                for (int w = 0; w < source.GetLength(1); w++)
                    for (int h = 0; h < source.GetLength(2); h++)
                        action(l,w, h);
        }
        public static void ForEach<T>(this T[] source, Action<int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
                action(w);
        }
    }
}
