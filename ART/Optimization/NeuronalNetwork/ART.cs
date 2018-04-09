using System;

namespace Optimization
{
  public class ART
  {

		#region Parameters

		#region Indexs

		//input index
		public int i { get; set; }

		//output index
		public int j { get; set; }

		//Winner neuron index
		public int J { get; set; }

		#endregion

		#region Matrix Sizes

		//input size
		public int M { get; set; }

		//Category matrix size
		public int N { get; set; }

		//input count list
		public int I { get; set; }

		//input count list
		public int CategoryRemaining { get; set; }

		#endregion

		#region Matrix Definitions

		//Matrix definitions of Inputs[i]
		public short[][] Inputs { get; set; }
		
		////Output neurons Y[j]
		//public short[] Y { get; set; }

		//Forward weight matrix. For each input the matrix contains weights to match input with output
		// W[i][j]
		public double[][] W { get; set; }

		//Feedback weight matrix
		//T[i][j]
		public double[][] T { get; set; }

		//Output neurons Z[j]
		public double[] Z { get; set; }

		//Inhibition layer to ignore 
		public short[] InhibitionLayer { get; set; }

		#endregion

		#region Vigilance

		public double p { get; set; }

		public double g { get; set; }

		#endregion

		#endregion

		#region Constructor
		public ART()
		{

		}

		#endregion


		public void Start()
		{
			InitParameters();

			for (int inputIndex = 0; inputIndex < I; inputIndex++)
			{
				ResetParameters();
				Recognize(inputIndex);

				Console.Write("Input index I: " + inputIndex + " Result J: " + J + " Input: ");
				foreach (var item in Inputs[inputIndex])
				{
					Console.Write(item + " ");
				}

				Console.Write(" Output: ");
				foreach (var item in Inputs[J])
				{
					Console.Write(item + " ");
				}
				
				Console.Write("\n");
			}
		}

		private void ResetParameters()
		{
			Z = new double[N];

			InhibitionLayer = new short[N];
			for (i = 0; i < N; i++)
			{
				InhibitionLayer[i] = 1;
			}

			CategoryRemaining = N;
		}

		private void Recognize(int inputIndex)
		{
			do
			{
				SetZVector(inputIndex);

				var zMax = GetZMaxAndSetJ();

				//Console.WriteLine("Z max: " + zMax);

				if (zMax > 0)
				{
					Comparer(inputIndex);
				}
				else
				{
					if (InhibitionLayer.Length > 0)
						InhibitionLayer[J] = 0;

					CategoryRemaining = 0;
					AddNewCategory(inputIndex);
				}
			}
			while (CategoryRemaining > 0);
		}

		private void Comparer(int a)
		{
			var x = SetForwardF0F1Transformation(a);

			var threashold = GetThreashold(a, x);

			//Console.WriteLine("Threashold: " + threashold + " P: " + p * p);

			if (threashold >= p * p)
			{
				ForwardPropagation(a);

				BackwardPropagation();

				CategoryRemaining = 0;
			}
			else
			{
				//Search();
				InhibitionLayer[J] = 0;
				CategoryRemaining--;
				if (CategoryRemaining <= 0)
				{
					AddNewCategory(a);
				}
				
			}
		}

		private void BackwardPropagation()
		{
			double wSum = 0;
			for (i = 0; i < M; i++)
			{
				wSum += W[J][i];
			}

			for (i = 0; i < M; i++)
			{
				T[J][i] = W[J][i] / (g + wSum);
			}
		}

		private void ForwardPropagation(int a)
		{
			for (i = 0; i < M; i++)
			{
				W[J][i] = Inputs[a][i] * W[J][i];
			}
		}

		private double GetThreashold(int a, double[] x)
		{
			double xNorm = 0;
			double inputNorm = 0;
			for (i = 0; i < M; i++)
			{
				xNorm += x[i] * x[i];
				inputNorm += Inputs[a][i] * Inputs[a][i];
			}

			return (xNorm / inputNorm);
		}

		private double[] SetForwardF0F1Transformation(int a)
		{
			var x = new double[M];
			for (i = 0; i < M; i++)
			{
				x[i] = (double)Inputs[a][i] * W[J][i];
			}

			return x;
		}

		private void AddNewCategory(int inputIndex)
		{
			N++;
			
			var w = W;
			Array.Resize(ref w, W.Length + 1);
			W = w;

			if (W[W.Length - 1] == null)
				W[W.Length - 1] = new double[M];
			
			J = W.Length - 1;
			for (int i = 0; i < M; i++)
			{
				W[W.Length - 1][i] = Inputs[inputIndex][i];
			}

			var t = T;
			Array.Resize(ref t, T.Length + 1);
			T = t;
			double wSum = 0;
			for (i = 0; i < M; i++)
			{
				wSum += W[J][i];
			}

			if (T[T.Length - 1] == null)
				T[T.Length - 1] = new double[M];

			for (i = 0; i < M; i++)
			{
				T[T.Length - 1][i] = (double)W[J][i] / (double)(g + wSum);
			}

			var z = Z;
			Array.Resize(ref z, Z.Length + 1);
			Z = z;

			var il = InhibitionLayer;
			Array.Resize(ref il, InhibitionLayer.Length + 1);
			InhibitionLayer = il;

		}

		private double GetZMaxAndSetJ()
		{
			double valueZMax = 0;
			for (j = 0; j < N; j++)
			{
				if ((Z[j] * InhibitionLayer[j]) > valueZMax)  //Falta inhibir las nueronas que ya salieron aplicando una and
				{
					valueZMax = Z[j];
					J = j;
				}
			}

			return valueZMax;
		}

		private void SetZVector(int a)
		{
			for (j = 0; j < N; j++)
			{
				Z[j] = 0;

				for (i = 0; i < M; i++)
				{
					Z[j] += Inputs[a][i] * W[j][i];
				}
			}
		}

		private void InitParameters()
		{
			//LoadInput();
			M = 4;
			I = 10;
			N = 0;

			Inputs = new short[I][];
			Inputs[0] = new short[] { 1, 0, 0, 0 };
			Inputs[1] = new short[] { 0, 0, 0, 1 };
			Inputs[2] = new short[] { 0, 0, 1, 0 };
			Inputs[3] = new short[] { 0, 0, 1, 1 };
			Inputs[4] = new short[] { 0, 1, 0, 0 };
			Inputs[5] = new short[] { 0, 1, 0, 1 };
			Inputs[6] = new short[] { 0, 1, 1, 0 };
			Inputs[7] = new short[] { 0, 1, 1, 1 };
			Inputs[8] = new short[] { 1, 0, 0, 0 };
			Inputs[9] = new short[] { 1, 0, 0, 1 };

			//I = 3;
			//Inputs[0] = new short[] { 1, 1, 0, 0 };
			//Inputs[1] = new short[] { 0, 0, 1, 1 };
			//Inputs[2] = new short[] { 1, 1, 1, 0 };


			//Initialize();
			p = 0.5;
			g = 0.5;

			//Setting Wij & Tij
			W = new double[N][];
			T = new double[N][];

			for (i = 0; i < M; i++)
			{
				for (j = 0; j < N; j++)
				{
					if (W[j] == null)
						W[j] = new double[M];

					if (T[j] == null)
						T[j] = new double[M];

					W[j][i] = (double)1 / (1 + M);
					T[j][i] = 1;
				}
			}

			
		}
	}
}
