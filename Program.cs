// See https://aka.ms/new-console-template for more information


using FirstML;

using Microsoft.ML;
//using RestSharp;

//var client = new RestClient("https://yh-finance.p.rapidapi.com/auto-complete?q=tesla&region=US");
//var request = new RestRequest(Method.GET);
//request.AddHeader("x-rapidapi-host", "yh-finance.p.rapidapi.com");
//request.AddHeader("x-rapidapi-key", "c89f412dfamsh3940017332859eep169dc5jsnab90764d8ac3");
//IRestResponse response = client.Execute(request);

try
{

   //string responseContent = response.Content;
   var context = new MLContext();

   var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

   var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

   var features = split.TrainSet.Schema
      .Select(col => col.Name)
      .Where(colName => colName != "Label" && colName != "OceanProximity")
      .ToArray();

   var pipeline = context.Transforms.Text
      .FeaturizeText("Text", "OceanProximity")
      //.Append(context.Transforms.Concatenate("Features", features))
      //.Append(context.Transforms.Concatenate("Features", "Features", "Text"))
      .Append(context.Regression.Trainers.LbfgsPoissonRegression());
   var model = pipeline.Fit(split.TrainSet);
   var predictions = model.Transform(split.TrainSet);
   var metrics = context.Regression.Evaluate(predictions);
   Console.WriteLine($"R^2 - {metrics.RSquared}");
}
catch (Exception e)
{
   Console.WriteLine(e);
}