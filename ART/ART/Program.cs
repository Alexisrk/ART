using System;

namespace ART
{
  class Program
  {
    static void Main(string[] args)
    {
      Console.WriteLine("Start");

			var art = new Optimization.ART();
			art.Start();

			Console.WriteLine("End");
			Console.Read();
		}

  }
}
