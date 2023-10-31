using OpenCvSharp;
using Sdcb.OpenVINO;
using Sdcb.OpenVINO.Extensions.OpenCvSharp4;
using Sdcb.OpenVINO.Natives;
using System.Diagnostics;

public class Program
{
    static unsafe void Main()
    {
        string modelFile = DownloadModel().GetAwaiter().GetResult();
        using VideoCapture vc = new(0);

        using Model rawModel = OVCore.Shared.ReadModel(modelFile);
        using PrePostProcessor pp = rawModel.CreatePrePostProcessor();
        using (PreProcessInputInfo inputInfo = pp.Inputs.Primary)
        {
            inputInfo.TensorInfo.Layout = Layout.NHWC;
            inputInfo.TensorInfo.ElementType = ov_element_type_e.U8;
            inputInfo.TensorInfo.SpatialStaticShape = (vc.FrameHeight, vc.FrameWidth);
            inputInfo.ModelInfo.Layout = Layout.NCHW;
            inputInfo.Steps.Resize(ov_preprocess_resize_algorithm_e.RESIZE_LINEAR);
        }
        using Model m = pp.BuildModel();
        using CompiledModel cm = OVCore.Shared.CompileModel(m, "CPU");
        using InferRequest ir = cm.CreateInferRequest();

        Shape modelInputSize = m.Inputs.Primary.Shape;
        Console.WriteLine(modelInputSize);
        while (vc.Grab())
        {
            using Mat frame = vc.RetrieveMat();
            Stopwatch sw = Stopwatch.StartNew();

            using (Tensor tensor = frame.AsTensor())
            {
                ir.Inputs.Primary = tensor;
            }
            double preprocessTime = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            ir.Run();
            double inferTime = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            using Tensor output = ir.Outputs.Primary;
            Shape outputShape = output.Shape;
            ReadOnlySpan<float> result = output.GetData<float>();

            List<DetectionResult> results = new();
            for (int i = 0; i < outputShape[2]; ++i)
            {
                float confidence = result[i * 7 + 2];
                int clsId = (int)result[i * 7 + 1];
                if (confidence > 0.5)
                {
                    int x1 = (int)(result[i * 7 + 3] * frame.Width);
                    int y1 = (int)(result[i * 7 + 4] * frame.Height);
                    int x2 = (int)(result[i * 7 + 5] * frame.Width);
                    int y2 = (int)(result[i * 7 + 6] * frame.Height);

                    results.Add(new(clsId, confidence, new Rect(x1, y1, x2 - x1, y2 - y1)));
                }
            }
            double postprocessTime = sw.Elapsed.TotalMilliseconds;

            double totalTime = preprocessTime + inferTime + postprocessTime;
            Console.WriteLine($"totalTime={totalTime:F2}ms");
            foreach (DetectionResult r in results)
            {
                Cv2.PutText(frame, $"{r.Confidence:P0}", r.Rect.TopLeft, HersheyFonts.HersheyPlain, 1, Scalar.Red);
                Cv2.Rectangle(frame, r.Rect, Scalar.Red);
            }
            Cv2.PutText(frame, $"Preprocess: {preprocessTime:F2}ms", new Point(10, 20), HersheyFonts.HersheyPlain, 1, Scalar.Red);
            Cv2.PutText(frame, $"Infer: {inferTime:F2}ms", new Point(10, 40), HersheyFonts.HersheyPlain, 1, Scalar.Red);
            Cv2.PutText(frame, $"Postprocess: {postprocessTime:F2}ms", new Point(10, 60), HersheyFonts.HersheyPlain, 1, Scalar.Red);
            Cv2.PutText(frame, $"Total: {totalTime:F2}ms", new Point(10, 80), HersheyFonts.HersheyPlain, 1, Scalar.Red);

            Cv2.ImShow("frame", frame);
            Cv2.WaitKey(1);
        }
    }

    static async Task<string> DownloadModel()
    {
        string rootFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "face-detection-0200");
        Directory.CreateDirectory(rootFolder);
        string modelFile = Path.Combine(rootFolder, "face-detection-0200.xml");
        if (!File.Exists(modelFile))
        {
            Console.WriteLine($"Downloading model to {rootFolder}...");
            string weightsFile = Path.Combine(rootFolder, "face-detection-0200.bin");

            using HttpClient http = new();
            File.WriteAllBytes(modelFile, await http.GetByteArrayAsync(@"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/2/face-detection-0200/FP16/face-detection-0200.xml"));
            File.WriteAllBytes(weightsFile, await http.GetByteArrayAsync(@"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/2/face-detection-0200/FP16/face-detection-0200.bin"));
        }
        else
        {
            Console.WriteLine($"Model already exists in {rootFolder}.");
        }

        return modelFile;
    }
}

public record DetectionResult(int ClassId, float Confidence, Rect Rect);