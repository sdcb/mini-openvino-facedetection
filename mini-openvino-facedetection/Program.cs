using OpenCvSharp;
using Sdcb.OpenVINO;
using System.Diagnostics;
using System.Runtime.InteropServices;

string modelFile = DownloadModel().GetAwaiter().GetResult();

using Model m = OVCore.Shared.ReadModel(modelFile);
using CompiledModel cm = OVCore.Shared.CompileModel(m, "CPU");
using InferRequest ir = cm.CreateInferRequest();

NCHW modelInputSize = m.Inputs.Primary.Shape.ToNCHW();
Console.WriteLine(modelInputSize);
using VideoCapture vc = new(0);
while (vc.Grab())
{
    using Mat frame = vc.RetrieveMat();
    Stopwatch sw = Stopwatch.StartNew();
    using Mat resized = frame.Resize(new Size(modelInputSize.Width, modelInputSize.Height));
    using Mat normalized = Normalize(resized);
    float[] extracted = ExtractMat(normalized);

    using (Tensor tensor = Tensor.FromArray(extracted, modelInputSize.ToShape()))
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
    Span<float> result = output.GetData<float>();
    
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
            
            results.Add(new (clsId, confidence, new Rect(x1, y1, x2-x1, y2 - y1)));
        }
    }
    double postprocessTime = sw.Elapsed.TotalMilliseconds;

    double totalTime = preprocessTime + inferTime + postprocessTime;
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

static float[] ExtractMat(Mat src)
{
    int rows = src.Rows;
    int cols = src.Cols;
    float[] result = new float[rows * cols * 3];
    GCHandle resultHandle = default;
    try
    {
        resultHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
        IntPtr resultPtr = resultHandle.AddrOfPinnedObject();
        for (int i = 0; i < src.Channels(); ++i)
        {
            using Mat dest = new(rows, cols, MatType.CV_32FC1, resultPtr + i * rows * cols * sizeof(float));
            Cv2.ExtractChannel(src, dest, i);
        }
    }
    finally
    {
        resultHandle.Free();
    }
    return result;
}

static Mat Normalize(Mat src)
{
    using Mat normalized = new();
    src.ConvertTo(normalized, MatType.CV_32FC3);
    Mat[] bgr = normalized.Split();
    float[] scales = new[] { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
    float[] means = new[] { 0.485f, 0.456f, 0.406f };
    for (int i = 0; i < bgr.Length; ++i)
    {
        bgr[i].ConvertTo(bgr[i], MatType.CV_32FC1, 1.0 * scales[i], (0.0 - means[i]) * scales[i]);
    }

    Mat dest = new();
    Cv2.Merge(bgr, dest);

    foreach (Mat channel in bgr)
    {
        channel.Dispose();
    }

    return dest;
}

async Task<string> DownloadModel()
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

public record DetectionResult(int ClassId, float Confidence, Rect Rect);