package agents.sergeyPolikarpov;

import java.util.Random;

/**
 * Created by Eclipse Java EE IDE.
 * User: Sergey V. Polikarpov
 * Date: Aug 29, 2009
 * Time: 16:09:00
 * Controller for marioai, based on CyberNeuron (see technical details at http://arxiv.org/abs/0907.0229)
 * Copyright (C) 2009  Sergey V. Polikarpov, Konstantin E. Rumyantsev
 * Contact: polikarpovsv*gmail.com
 * <p>
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

public class CyberNeuron {

    private float[][][] sbox;

    private int[] inputs_for_sbox;
    private int[] outputs_for_sbox;
    private double[] outputs;
    private double[] inputs;
    public final int parallel_inputs = 1;

    private int num_bits_in_input = 10; //****//
    private int num_of_cells_in_sbox = (int) Math.pow(2, num_bits_in_input); //****//
    private int[] powstwo = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

    private int threshold;

    private final Random random = new Random();
    private double learningRate = 0.05;

    public CyberNeuron(int numberOfInputs, int numberOfOutputs) {
        threshold = (int) (0.65 * (127 * numberOfInputs * parallel_inputs / num_bits_in_input));
        sbox = new float[numberOfInputs * parallel_inputs / num_bits_in_input][num_of_cells_in_sbox][numberOfOutputs];   //****//
        inputs_for_sbox = new int[sbox.length];
        outputs_for_sbox = new int[sbox[0][0].length];

        outputs = new double[numberOfOutputs];
        inputs = new double[numberOfInputs];
        initialize_sbox(sbox);
    }

    public CyberNeuron(float[][][] sbox, int numberOfOutputs) {
        this.sbox = sbox;
        inputs = new double[sbox.length * num_bits_in_input];
        outputs = new double[numberOfOutputs];
    }

    protected void initialize_sbox(float[][][] sbox) {
        for (int i = 0; i < sbox.length; i++) {
            for (int j = 0; j < sbox[i].length; j++) {
                for (int m = 0; m < sbox[i][j].length; m++) {
                    sbox[i][j][m] = 0;
                }
            }
        }
    }


    public CyberNeuron getNewInstance() {
        return new CyberNeuron(sbox.length * num_bits_in_input / parallel_inputs, outputs.length);
    }

    public CyberNeuron copy() {
        CyberNeuron copy = new CyberNeuron(copy(sbox), outputs.length);
        //copy.setMutationMagnitude(mutationMagnitude);
        return copy;
    }

    private float[][][] copy(float[][][] original) {  // need to correct
        float[][][] copy = new float[original.length][original[0].length][original[0][0].length];
        for (int i = 0; i < copy.length; i++) {
            for (int j = 0; j < copy[i].length; j++) {
                for (int m = 0; m < copy[i][j].length; m++) {
                    copy[i][j][m] = original[i][j][m];
                }
            }
        }
        return copy;
    }

    public double[] propagate(double[] inputIn) {
        if (inputs != inputIn) {
            System.arraycopy(inputIn, 0, this.inputs, 0, inputIn.length);
        }
        if (inputIn.length < inputs.length)
            System.out.println("NOTE: only " + inputIn.length + " inputs out of " + inputs.length + " are used in the network");

        int blocksize = sbox.length / parallel_inputs;
        int[] tmp_inputs = new int[blocksize];
        for (int m = 0; m < blocksize; m++) {
            tmp_inputs[m] = 0;
            for (int n = 0; n < num_bits_in_input; n++) {
                if (inputs[m * num_bits_in_input + n] == 1) {
                    tmp_inputs[m] += powstwo[n];
                }
            }
        }

        for (int p = 0; p < parallel_inputs; p++) {
            for (int m = 0; m < blocksize; m++) {                                   //****//
                inputs_for_sbox[p * blocksize + m] = tmp_inputs[m];
            }
        }

        for (int i = 0; i < sbox[0][0].length; i++) {                                //****//
            outputs_for_sbox[i] = 0;
            for (int m = 0; m < sbox.length; m++) {
                outputs_for_sbox[i] += sbox[m][inputs_for_sbox[m]][i];
            }

            double tmp_value;
            if (outputs_for_sbox[i] > this.threshold) {
                tmp_value = this.threshold;
            } else {
                tmp_value = outputs_for_sbox[i];
            }

            outputs[i] = tmp_value / (double) this.threshold;
        }
        return outputs;
    }


    public double backPropagate(double[] targetOutputs) {
        double[] outputError = new double[outputs.length];

        for (int i = 0; i < targetOutputs.length; i++) {
            /*if(targetOutputs[i] == 1) {
            	if(outputs[i]*this.threshold < this.threshold) {outputError[i] = (targetOutputs[i] - outputs[i])*this.threshold;}
            	else {outputError[i] = 0;}
            }
            if(targetOutputs[i] == 0) {
            	if(outputs[i]*this.threshold < this.threshold_min) {outputError[i] = 0;}
            	else {outputError[i] = (targetOutputs[i] - outputs[i])*this.threshold;}
            }*/
            outputError[i] = (targetOutputs[i] - outputs[i]) * this.threshold;

            if (Double.isNaN(outputError[i])) {
                System.out.println("Problem at output " + i);
                System.out.println(outputs[i] + " " + targetOutputs[i]);
                System.exit(0);
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //CELLS UPDATE
        ///////////////////////////////////////////////////////////////////////////
        for (int i = 0; i < sbox[0][0].length; i++) {
            int outputError_tmp = (int) (outputError[i] * learningRate);
            if (outputError_tmp == 0) {
                if (outputError[i] > 0) {
                    outputError[i] = 1;
                }
                if (outputError[i] < 0) {
                    outputError[i] = -1;
                }
                if (outputError[i] == 0) {
                    outputError[i] = 0;
                }
            } else {
                outputError[i] = outputError_tmp;
            }

            if (outputError[i] > 0) {
                for (int r = 0; r < outputError[i]; r++) {
                    int row_number = random.nextInt(sbox.length);
                    int cell_number = inputs_for_sbox[row_number];
                    //if(sbox[row_number][cell_number][i] < 126)
                    {
                        sbox[row_number][cell_number][i]++;
                    }
                }

            } else {
                for (int r = 0; r < Math.abs(outputError[i]); r++) {
                    int row_number = random.nextInt(sbox.length);
                    int cell_number = inputs_for_sbox[row_number];
                    //if(sbox[row_number][cell_number][i] > -125)
                    {
                        sbox[row_number][cell_number][i]--;
                    }
                }
            }

        }

        double summedOutputError = 0.0;
        for (int k = 0; k < targetOutputs.length; k++) {
            summedOutputError += Math.abs(targetOutputs[k] - outputs[k]);
        }
        summedOutputError /= outputs.length;

        // Return something sensible
        return summedOutputError;
    }

    public double getMutationMagnitude() {
        return 0;
    }

    public void setMutationMagnitude(double mutationMagnitude) {
        //this.mutationMagnitude = mutationMagnitude;
    }

    public static void setInitParameters(double mean, double deviation) {
        System.out.println("PARAMETERS SET: " + mean + "  deviation: " + deviation);
    }

    public void println() {
        System.out.print("\n\n----------------------------------------------------" +
                "-----------------------------------\n");
        System.out.print("----------------------------------------------------" +
                "-----------------------------------\n");
    }

    public String toString() {
        return "CyberNeuron by Sergey V. Polikarpov";
    }

    public void ssetLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double[] getOutputs() {
        double[] outputsCopy = new double[outputs.length];
        System.arraycopy(outputs, 0, outputsCopy, 0, outputs.length);
        return outputsCopy;
    }

    public double[] getWeightsArray() {

        double[] weights = new double[sbox.length * sbox[0].length * sbox[0][0].length];

        int k = 0;
        for (int i = 0; i < sbox.length; i++) {
            for (int j = 0; j < sbox[i].length; j++) {
                for (int m = 0; m < sbox[i][j].length; m++) {
                    k = i + j + m;
                    weights[k] = sbox[i][j][m];
                }
            }
        }
        return weights;
    }

    public void setWeightsArray(double[] weights) {
        int k = 0;

        for (int i = 0; i < sbox.length; i++) {
            for (int j = 0; j < sbox[i].length; j++) {
                for (int m = 0; m < sbox[i][j].length; m++) {
                    k = i + j + m;
                    sbox[i][j][m] = (char) weights[k];
                }
            }
        }
    }

    public int getNumberOfInputs() {
        return inputs.length;
    }

    public void randomise() {
    }

    public double[] getArray() {
        return getWeightsArray();
    }

    public void setArray(double[] array) {
        setWeightsArray(array);
    }

    public int[] getPackedInputsToInt(double[] inputs) {
        int[] tmp_inputs = new int[sbox.length];
        for (int m = 0; m < sbox.length; m++) {
            tmp_inputs[m] = 0;
            for (int n = 0; n < num_bits_in_input; n++) {
                if (inputs[m * num_bits_in_input + n] == 1) {
                    tmp_inputs[m] += powstwo[n];
                }
            }
        }
        return tmp_inputs;
    }

}
