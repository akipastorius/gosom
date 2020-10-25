/*
Copyright © 2020 NAME HERE <EMAIL ADDRESS>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package cmd

import (
	"fmt"
	"math/rand"

	"aki.ali-vehmas/som/internal/som"
	"github.com/spf13/cobra"
)

// runCmd represents the run command
var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Train the som based on data",
	Long: `Train the self-organizing map based on data (csv)
	Self-organizing map maps the input data into two-dimensional
	topology-preserving space by using competitive learning`,
	Run: func(cmd *cobra.Command, args []string) {
		x, _ := cmd.Flags().GetInt("x")
		y, _ := cmd.Flags().GetInt("y")
		maxIter, _ := cmd.Flags().GetInt("maxIter")
		seed, _ := cmd.Flags().GetInt("seed")
		inputFilePath, _ := cmd.Flags().GetString("inputFilePath")
		verbose, _ := cmd.Flags().GetBool("verbose")
		initZeros, _ := cmd.Flags().GetBool("initZeros")

		rand.Seed(int64(seed))
		data, _ := som.ReadData(inputFilePath)
		_, n := data.Dims()

		somap := som.NewSom(x, y, n, initZeros)

		som.Train(somap, data, maxIter, verbose)
		som.WriteData(somap, data, "data/result.csv")
		if verbose {
			fmt.Println("final weights:")
			som.PrintSomWeights(somap)
		}

	},
}

func init() {
	rootCmd.AddCommand(runCmd)
	runCmd.Flags().Int("x", 2, "Map x dimension")
	runCmd.Flags().Int("y", 2, "Map y dimension")
	runCmd.Flags().Bool("initZeros", false, "Initialize map with zeros")
	runCmd.Flags().Int("maxIter", 1000, "Maximum iterations")
	runCmd.Flags().Int("seed", 0, "Random seed")
	runCmd.Flags().String("inputFilePath", "data/data.csv", "Input data filepath")
	runCmd.Flags().Bool("verbose", false, "Toggle verbose mode")

	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// runCmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	// runCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}
