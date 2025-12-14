using System.Collections.Generic;
using UnityEngine;

public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff()
    {
        return coeficients;
    }
    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }
    public List<float[]> GetInter()
    {
        return intercepts;
    }
}

public class MLPModel
{
    MLPParameters mlpParameters;
    public MLPModel(MLPParameters p)
    {
        mlpParameters = p;
    }

    /// <summary>
    /// Parameters required for model input. By default it will be perception, kart position and time, 
    /// but depending on the data cleaning and data acquisition modificiations made by each one, the input will need more parameters.
    /// </summary>
    /// <param name="p">The Agent perception</param>
    /// <returns>The action label</returns>
    public float[] FeedForward(float[] input)
    {
        List<float[,]> Ws = mlpParameters.GetCoeff();
        List<float[]> Bs = mlpParameters.GetInter();

        float[] a = input;

        for (int layer = 0; layer < Ws.Count; layer++)
        {
            float[,] W = Ws[layer];
            float[] b = Bs[layer];

            int inSize = W.GetLength(0);
            int outSize = W.GetLength(1);

            if (a.Length != inSize)
            {
                Debug.LogError($"MLP FeedForward: el tamaño del input {a.Length} != " +
                    $"esperado {inSize} en la capa {layer}");
                return new float[outSize];
            }

            float[] z = new float[outSize];
            for (int j = 0; j < outSize; j++)
            {
                float sum = b[j];
                for (int i = 0; i < inSize; i++)
                {
                    sum += a[i] * W[i, j];
                }
                z[j] = sum;
            }

            bool lastLayer = (layer == Ws.Count - 1);
            if (lastLayer) a = SoftMax(z);
            else
            {
                for (int j = 0; j < z.Length; j++)
                {
                    z[j] = sigmoid(z[j]);
                }
                a = z;
            }
        }

        return a;
    }

    /// <summary>
    /// Calculo de la sigmoidal
    /// </summary>
    /// <param name="z"></param>
    /// <returns></returns>
    private float sigmoid(float z)
    {
        // Forma estable
        if (z >= 0f)
        {
            float e = Mathf.Exp(-z);
            return 1f / (1f + e);
        }
        else
        {
            float e = Mathf.Exp(z);
            return e / (1f + e);
        }
    }


    /// <summary>
    /// Cálculo de la soft max, se le pasa el vector de la ulrima capa oculta y devuelve el mismo vector, pero procesado
    /// aplicando softmax a cada uno de los elementos
    /// </summary>
    /// <param name="zArr"></param>
    /// <returns></returns>
    public float[] SoftMax(float[] zArr)
    {
        float max = zArr[0];
        for (int i = 0; i < zArr.Length; i++)
        {
            max = Mathf.Max(max, zArr[i]);
        }

        float sum = 0f;
        float[] exp = new float[zArr.Length];
        for (int i = 0; i < zArr.Length; i++)
        {
            exp[i] = Mathf.Exp(zArr[i] - max);
            sum += exp[i];
        }

        for (int i = 0; i < zArr.Length; i++)
        {
            exp[i] /= sum;
        }

        return exp;
    }

    /// <summary>
    /// Elige el output de mayor nivel
    /// </summary>
    /// <param name="output"></param>
    /// <returns></returns>
    public int Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        return index;
    }

    /// <summary>
    /// Obtiene el índice de mayor valor.
    /// </summary>
    /// <param name="output"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;
        for (int i = 1; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
}
