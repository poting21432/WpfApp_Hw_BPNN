using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using NN_MNIST;
namespace WpfApp_BP_NN_handwrite_recognition
{
    /// <summary>
    /// MainWindow.xaml 的互動邏輯
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Global.RichTextBox_Log = this.RichTextBox_Log;
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Global.ImagesTrain = MnistReader.ReadTrainingData().ToList();
            Global.ImagesTest = MnistReader.ReadTestData().ToList();
            TextBox_ImageTrainID_TextChanged(TextBox_ImageTrainID, null);
            TextBox_ImageTestID_TextChanged(TextBox_ImageTestID, null);
        }

        private void Button_Train_Click(object sender, RoutedEventArgs e)
        {
            Global.BP.Train(Global.ImagesTrain);
        }

        private void Button_Test_Click(object sender, RoutedEventArgs e)
        {
            Global.BP.Test( Global.ImagesTest);
        }

        private void TextBox_ImageTrainID_TextChanged(object sender, TextChangedEventArgs e)
        {
            int.TryParse(TextBox_ImageTrainID.Text, out int result);
            if (Global.ImagesTrain == null)
                return;
            if (result >= Global.TrainSize)
            {
                result = Global.TrainSize - 1;
                TextBox_ImageTrainID.Text = result.ToString();
                return;
            }
            
            var image = Global.ImagesTrain[result];
            Image_Train.Source =
            WpfHelper.BitmapToImageSource(
                WpfHelper.Array2DToBitmap(image.Data, Global.ImageWidth, Global.ImageHeight));
            TextBlox_TrainLabel.Text = image.Label.ToString();
            TextBlox_TrainPredict.Text = "Predict: " + Global.BP.TestOne(Global.ImagesTrain, result).ToString();
        }

        private void TextBox_ImageTestID_TextChanged(object sender, TextChangedEventArgs e)
        {
            int.TryParse(TextBox_ImageTestID.Text, out int result);
            if (Global.ImagesTest == null)
                return;

            if (result >= Global.TestSize)
            {
                result = Global.TestSize - 1;
                TextBox_ImageTestID.Text = result.ToString();
                return;
            }

            var image = Global.ImagesTest[result];
            Image_Test.Source =
            WpfHelper.BitmapToImageSource(
                WpfHelper.Array2DToBitmap(Global.ImagesTest[result].Data, Global.ImageWidth, Global.ImageHeight));
            TextBlox_TestLabel.Text = "Label: "+ image.Label.ToString();
            TextBlox_TestPredict.Text = "Predict: " + Global.BP.TestOne(Global.ImagesTest, result).ToString();
        }
        private void Button_Increase_Click(object sender, RoutedEventArgs e)
        {
            TextBox textBox = (TextBox)((Control)(sender)).Tag;
            int.TryParse(textBox.Text, out int result);
            result++;
            textBox.Text = result.ToString();
        }

        private void Button_Decrease_Click(object sender, RoutedEventArgs e)
        {
            TextBox textBox = (TextBox)((Control)(sender)).Tag;
            int.TryParse(textBox.Text, out int result);
            result--;
            if (result < 0)
                result = 0;
            textBox.Text = result.ToString();
        }

        private void ButtonStop_Click(object sender, RoutedEventArgs e)
        {
            Global.BP.StopSig = true;
        }

        private void Button_OutputFile_Click(object sender, RoutedEventArgs e)
        {
            string ans ="";
            int cnt = 1;
            foreach(var img in Global.ImagesTest)
            {
                ans += string.Format("{0} {1}\r\n", cnt.ToString("D4"), img.Predict);
                cnt++;
            }
            if (File.Exists("./Answer.txt"))
                File.Delete("./Answer.txt");
            File.WriteAllText("./Answer.txt", ans);
        }
    }
}
