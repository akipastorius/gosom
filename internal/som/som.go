package som

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/kniren/gota/dataframe"
	"gonum.org/v1/gonum/mat"
)

type som struct {
	x, y, n      int
	weights      []mat.Dense
	sigma        float64
	learningRate float64
	neigx        mat.Vector
	neigy        mat.Vector
}

type matrix struct {
	dataframe.DataFrame
}

func (m matrix) At(i, j int) float64 {
	return m.Elem(i, j).Float()
}

func (m matrix) T() mat.Matrix {
	return mat.Transpose{Matrix: m}
}

// NewSom creates new som
func NewSom(x, y, n int, initZero bool) *som {
	som := som{
		x: x,
		y: y,
		n: n,
	}
	som = som.initWeights(initZero)
	som.sigma = 1.0
	som.learningRate = 1.0
	var neigx []float64
	var neigy []float64
	for i := 0; i < som.x; i++ {
		neigx = append(neigx, float64(i))
	}

	for i := 0; i < som.y; i++ {
		neigy = append(neigy, float64(i))
	}

	som.neigx = mat.NewVecDense(som.x, neigx)
	som.neigy = mat.NewVecDense(som.y, neigy)
	return &som
}

func (som *som) initWeights(zeros bool) som {
	for i := 0; i < som.n; i++ {
		d := make([]float64, som.x*som.y)
		for j := range d {
			if zeros {
				d[j] = 0.0
			} else {
				d[j] = rand.Float64()
			}

		}
		som.weights = append(som.weights, *mat.NewDense(som.x, som.y, d))
	}
	return *som
}

// PrintSomWeights prints weight vector
func PrintSomWeights(som *som) {
	for i := 0; i < som.n; i++ {
		fmt.Println("---")
		matPrint(som.weights[i])
	}
	fmt.Println("---")
}

func matPrint(X mat.Dense) {
	fa := mat.Formatted(&X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func distance(x float64, y float64) float64 {
	return math.Pow(x-y, 2)
}

func bestMatchingUnit(som *som, datapoint mat.Vector) (int, int) {
	x, y := -1, -1
	bestValue := math.Inf(1)
	for i := 0; i < som.x; i++ {
		for j := 0; j < som.y; j++ {
			value := 0.0
			for k := 0; k < som.n; k++ {
				value += distance(som.weights[k].At(i, j), datapoint.At(k, 0))
			}
			if value < bestValue {
				bestValue = value
				x = i
				y = j
			}
		}
	}
	return x, y
}

func decayFunction(value float64, t int, maxIter int) float64 {
	return value * math.Exp(0.05*((float64(t)/float64(maxIter))*100)*-1)
}

func neighborhoodFunction(som *som, x int, y int, sigma float64, learningRate float64) mat.Dense {
	d := 2 * math.Pi * sigma * sigma
	ax := mat.NewVecDense(som.x, nil)
	ay := mat.NewVecDense(som.y, nil)
	var value float64
	for i := 0; i < som.x; i++ {
		value = math.Exp((math.Pow(som.neigx.AtVec(i)-float64(x), 2) / d) * -1)
		ax.SetVec(i, value)
	}

	for i := 0; i < som.y; i++ {
		value = math.Exp((math.Pow(som.neigy.AtVec(i)-float64(y), 2) / d) * -1)
		ay.SetVec(i, value)
	}

	var m mat.Dense
	m.Outer(1, ax, ay)

	for i := 0; i < som.x; i++ {
		for j := 0; j < som.y; j++ {
			m.Set(i, j, m.At(i, j)*learningRate)
		}
	}

	return m

}

func updateWeights(som *som, datapoint mat.Vector, t int, maxIter int) {
	bmux, bmuy := bestMatchingUnit(som, datapoint)
	sig := decayFunction(som.sigma, t, maxIter)
	rate := decayFunction(som.learningRate, t, maxIter)
	nbhood := neighborhoodFunction(som, bmux, bmuy, sig, rate)

	for i := 0; i < som.x; i++ {
		for j := 0; j < som.y; j++ {
			for k := 0; k < som.n; k++ {
				som.weights[k].Set(i, j, som.weights[k].At(i, j)+(datapoint.AtVec(k)-som.weights[k].At(i, j))*nbhood.At(i, j))
			}
		}
	}
}

// Train trains som
func Train(som *som, data *mat.Dense, maxIter int, verbose bool) {
	start := time.Now()
	n, _ := data.Dims()
	chunk := maxIter / 10
	for t := 0; t < maxIter; t++ {
		if (math.Mod(float64(t), float64(chunk)) == 0) && verbose {
			elapsed := time.Since(start)
			log.Printf("iteration %v / %v - %v iterations / sec", t, maxIter, math.Round(1000/elapsed.Seconds()))
		}
		index := int(math.Mod(float64(t), float64(n)))

		updateWeights(som, data.RowView(index), t, maxIter)
	}
	elapsed := time.Since(start)
	if verbose {
		log.Printf("training took %f seconds", elapsed.Seconds())
	}

}

// ReadData reads data from csv to gonum mat
func ReadData(filepath string) (*mat.Dense, error) {
	csvfile, err := os.Open(filepath)
	if err != nil {
		log.Fatal(err)
	}

	df := dataframe.ReadCSV(csvfile, dataframe.HasHeader(false))
	mat := mat.DenseCopyOf(matrix{df})
	return mat, err
}

// WriteData ...
func WriteData(som *som, data *mat.Dense, filepath string) {
	f, _ := os.Create(filepath)
	defer f.Close()
	n, _ := data.Dims()
	resultMap := make([]int, n)
	for i := 0; i < n; i++ {
		datapoint := data.RowView(i)
		x, y := bestMatchingUnit(som, datapoint)
		resultMap[i] = x + y*(som.x+1-1)
		f.WriteString(strconv.Itoa(resultMap[i]) + "\n")
	}

}
