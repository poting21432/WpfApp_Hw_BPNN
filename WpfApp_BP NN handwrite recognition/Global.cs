using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using NN_MNIST;

namespace WpfApp_BP_NN_handwrite_recognition
{
    using Image = NN_MNIST.Image;
    public static class Global
    {
        public static BP BP = new BP();
        public const int ImageWidth = 28;
        public const int ImageHeight = 28;
        public static int TrainSize { get { return ImagesTrain?.Count ?? 0; } }
        public static int TestSize { get { return ImagesTest?.Count ?? 0; } }
        public static List<Image> ImagesTrain;
        public static List<Image> ImagesTest;
        public static RichTextBox RichTextBox_Log;
        public const int MaxLogCount = 500;
        public static int LogCount = 0;
        public static void AppendLog(string Log)
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                if(LogCount > MaxLogCount)
                    RichTextBox_Log.Document.Blocks.Remove(RichTextBox_Log.Document.Blocks.FirstBlock); //First line
                RichTextBox_Log.AppendText(Log);
                RichTextBox_Log.ScrollToEnd();
                LogCount++;
            });
        }
    }

    public static class WpfHelper
    {
        public static Bitmap Array2DToBitmap(byte[,] data, int Width, int Height)
        {
            unsafe
            {
                fixed (byte* ptr = data)
                {
                    IntPtr scan0 = new IntPtr(ptr);
                    Bitmap bitmap = new Bitmap(Width, Height, // Image size
                                               Width, // Scan size
                                               PixelFormat.Format8bppIndexed, scan0);
                    ColorPalette palette = bitmap.Palette;
                    palette.Entries[0] = Color.Black;
                    for (int i = 1; i < 256; i++)
                    {
                        palette.Entries[i] = Color.FromArgb((i * 7) % 256, (i * 7) % 256, 255);
                    }
                    bitmap.Palette = palette;

                    return bitmap;
                }
            }
        }
        public static BitmapImage BitmapToImageSource(Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Bmp);
                memory.Position = 0;
                BitmapImage bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();

                return bitmapimage;
            }
        }
    }
}
