package agents.sergeyPolikarpov;

import java.util.Random;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.MarioActions;

/**
 * Created by IntelliJ IDEA.
 * User: julian
 * Date: Apr 28, 2009
 * Time: 2:09:42 PM
 * <p>
 * <p>
 * Modified by Sergey V. Polikarpov
 * Date: Aug 29, 2009
 * Time: 16:09:00
 */
public class Agent implements MarioAgent {

    private CyberNeuron cbrn;
    private final int numberOfOutputs = 10;
    private final int block_size = 340;
    private final int numberOfInputs = 2 * block_size + numberOfOutputs;

    private final double LearningRate = 0.01;
    private boolean temporary_disable_cbrn = false;
    private boolean action_in_progress = false;
    private int action_in_progress_type;
    private int count_of_action_in_progress = 0;
    private final int action_forward = 1;
    private final int action_forward_jump = 2;
    private final int action_backward = 3;
    private final int action_backward_jump = 4;
    private final int action_forward_with_prone = 5;
    private final int action_forward_with_jump_far = 6;
    private final int action_none_action = 7;
    private int[][] progressed_action = new int[][]{{3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6}, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, {1, 1, 1, 1, 1, 1, 1, 1}, {3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6}, {2, 2, 2, 2, 2, 2}, {3, 3, 3, 3, 3}, {4, 4, 4, 4, 4, 4}, {1, 1, 1, 1, 1}};

    private int deep_of_buffer = 35;//16;
    private int num_pushes_to_buffer = 0;
    private boolean buffer_is_full = false;
    private double[][] buffer_of_inputs = new double[deep_of_buffer][numberOfInputs];
    private double[] buffer_of_actions = new double[deep_of_buffer];

    private int deep_of_buffer_of_Mario_X = 46;//16
    private int waiting_counter = 0;
    private int num_pushes_to_buffer_of_Mario_X = 0;
    private boolean buffer_is_full_of_Mario_X = false;
    private float[] buffer_of_Mario_X = new float[deep_of_buffer_of_Mario_X];
    private int prev_mario_mode = 2;

    private CyberNeuron detector_of_holes;
    private final int detector_of_holes_numberOfInputs = 22 * 10;
    private final int detector_of_holes_numberOfOutputs = 1;
    private final int detector_of_holes_deep_of_buffer = 17;//15;
    private boolean detector_of_holes_buffer_is_full = false;
    private int detector_of_holes_num_pushes_to_buffer = 0;
    private double[][] detector_of_holes_buffer_of_inputs = new double[detector_of_holes_deep_of_buffer][detector_of_holes_numberOfInputs];
    private boolean force_long_jump_forward = false;
    private boolean is_first_action_when_hole_is_detected = true;

    private static final Random random = new Random();

    /**********/

    private double probe(int x, int y, int[][] scene) {
        int realX = x + 8;
        int realY = y + 8;
        return (scene[realX][realY] != 0) ? 1 : 0;
    }

    private double probe_enemies(int x, int y, int[][] scene) {
        int realX = x + 8;
        int realY = y + 8;
        double result = 0;
        if (scene[realX][realY] != 1) {
            result = 1;
        } else {
            result = 0;
        }
        return result;
    }

    private void push_Mario_X_pos_to_buffer(float Mario_X_pos) {
        for (int i = 0; i < buffer_of_Mario_X.length - 1; i++) {
            buffer_of_Mario_X[i] = buffer_of_Mario_X[i + 1];
        }
        buffer_of_Mario_X[buffer_of_Mario_X.length - 1] = Mario_X_pos;
        num_pushes_to_buffer_of_Mario_X++;
        if (num_pushes_to_buffer_of_Mario_X >= deep_of_buffer_of_Mario_X) {
            buffer_is_full_of_Mario_X = true;
        }
    }

    private void push_inputs_and_actions_to_buffer(double[] inputs, int action_in_progress_type) {
        for (int i = 0; i < buffer_of_inputs.length - 1; i++) {
            for (int j = 0; j < buffer_of_inputs[i].length; j++) {
                buffer_of_inputs[i][j] = buffer_of_inputs[i + 1][j];
            }
        }
        for (int i = 0; i < inputs.length; i++) {
            buffer_of_inputs[buffer_of_inputs.length - 1][i] = inputs[i];
        }
        for (int i = 0; i < buffer_of_actions.length - 1; i++) {
            buffer_of_actions[i] = buffer_of_actions[i + 1];
        }
        buffer_of_actions[buffer_of_actions.length - 1] = action_in_progress_type;

        num_pushes_to_buffer++;
        if (num_pushes_to_buffer >= deep_of_buffer) {
            buffer_is_full = true;
        }
    }

    private void detector_of_holes_push_inputs_to_buffer(double[] inputs) {
        for (int i = 0; i < detector_of_holes_buffer_of_inputs.length - 1; i++) {
            for (int j = 0; j < detector_of_holes_buffer_of_inputs[i].length; j++) {
                detector_of_holes_buffer_of_inputs[i][j] = detector_of_holes_buffer_of_inputs[i + 1][j];
            }
        }
        for (int i = 0; i < inputs.length; i++) {
            detector_of_holes_buffer_of_inputs[detector_of_holes_buffer_of_inputs.length - 1][i] = inputs[i];
        }

        detector_of_holes_num_pushes_to_buffer++;
        if (detector_of_holes_num_pushes_to_buffer >= detector_of_holes_deep_of_buffer) {
            detector_of_holes_buffer_is_full = true;
        }
    }

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        cbrn = new CyberNeuron(numberOfInputs, numberOfOutputs);
        detector_of_holes = new CyberNeuron(detector_of_holes_numberOfInputs, detector_of_holes_numberOfOutputs);
        action_in_progress = false;
        count_of_action_in_progress = 0;
        buffer_is_full = false;
        num_pushes_to_buffer = 0;
        waiting_counter = 0;
        num_pushes_to_buffer_of_Mario_X = 0;
        buffer_is_full_of_Mario_X = false;
        prev_mario_mode = 2;
        detector_of_holes_buffer_is_full = false;
        detector_of_holes_num_pushes_to_buffer = 0;
        is_first_action_when_hole_is_detected = true;
        for (int i = 0; i < buffer_of_inputs.length; i++) {
            for (int j = 0; j < buffer_of_inputs[i].length; j++) {
                buffer_of_inputs[i][j] = 0;
            }
        }
        for (int i = 0; i < buffer_of_actions.length; i++) {
            buffer_of_actions[i] = 0;
        }
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        boolean[] action = new boolean[MarioActions.numberOfActions()];
        int[][] scene = model.getMarioSceneObservation();
        int[][] enemies = model.getMarioEnemiesObservation(2);
        float[] Mario_pos = model.getMarioFloatPos();
        double[] inputs = new double[numberOfInputs];
        int[] outs_of_cbrn = new int[numberOfOutputs];

        int which = 0;

        for (int f = 0; f < block_size / numberOfOutputs + 1; f++) {
            for (int i = 0; i < numberOfOutputs; i++) {
                if (i == buffer_of_actions[buffer_of_actions.length - 1 - f]) {
                    outs_of_cbrn[i] = 1;
                } else {
                    outs_of_cbrn[i] = 0;
                }
            }
            for (int i = 0; i < numberOfOutputs; i++) {
                inputs[which++] = outs_of_cbrn[i];
            }
        }

        for (int i = -6; i < 7; i++) {
            for (int j = -6; j < 7; j++) {
                inputs[which++] = probe(i, j, scene);
            }
        }
        for (int i = -6; i < 7; i++) {
            for (int j = -6; j < 7; j++) {
                inputs[which++] = probe_enemies(i, j, enemies);
            }
        }
        inputs[which++] = model.isMarioOnGround() ? 1 : 0;
        inputs[which++] = model.mayMarioJump() ? 1 : 0;

        /********* start detector_of_holes *******/
        int which2 = 0;
        double[] detector_of_holes_inputs = new double[detector_of_holes_numberOfInputs];
        for (int i = 0; i < 8; i++) {
            for (int j = -8; j < 8; j++) {
                detector_of_holes_inputs[which2++] = probe(i, j, scene);
            }
        }
        double[] detector_of_holes_result = detector_of_holes.propagate(detector_of_holes_inputs);
        int hole_is_close = 0;
        if (detector_of_holes_result[0] > 0.75) {
            if (Mario_pos[0] < 70) {
                hole_is_close = 0;
            } else {
                hole_is_close = 1;
            }
        }
        if (hole_is_close == 1) {
            action_in_progress = false;
            force_long_jump_forward = true;
            is_first_action_when_hole_is_detected = true;
        }
        /********* end detector_of_holes *******/

        if ((action_in_progress_type == 0) || (action_in_progress_type == 1) || (action_in_progress_type == 3))
            if ((model.isMarioOnGround() == true)/* &&(observation.mayMarioJump() == true) */) {
                if (count_of_action_in_progress > 3) {
                    action_in_progress = false;
                }
            }

        /****** Select sequence of actions ***********************/
        if (force_long_jump_forward == false) {
            if ((action_in_progress == false)
                    && (model.isMarioOnGround() == true)/* &&(observation.mayMarioJump() == true)) */) {
                double[] cbrn_action_in_progress = cbrn.propagate(inputs);
                boolean agent_is_active = false;
                int count_tmp = 0;
                for (int r = 0; r < cbrn_action_in_progress.length; r++) {
                    if (cbrn_action_in_progress[r] > 0.65) {
                        count_tmp++;
                    }
                }
                agent_is_active = count_tmp == 1;
                if (temporary_disable_cbrn == true) {
                    agent_is_active = false;
                }
                if (agent_is_active == true) {
                    for (int r = 0; r < cbrn_action_in_progress.length; r++) {
                        if (cbrn_action_in_progress[r] > 0.65) {
                            action_in_progress_type = r;
                        }
                    }
                }
                if (agent_is_active == false) { /************ if cyberneurons are inactive *****************/
                    int rvalue = random.nextInt(100);
                    if (rvalue < 30) {
                        action_in_progress_type = 0;
                    } else if (rvalue >= 30 && rvalue < 40) {
                        action_in_progress_type = 5;
                    } else if (rvalue >= 40 && rvalue < 50) {
                        action_in_progress_type = 1;
                    } else if (rvalue >= 50 && rvalue < 60) {
                        action_in_progress_type = 2;
                    } else if (rvalue >= 60 && rvalue < 70) {
                        action_in_progress_type = 3;
                    } else if (rvalue >= 70 && rvalue < 80) {
                        action_in_progress_type = 4;
                    } else if (rvalue >= 80 && rvalue < 85) {
                        action_in_progress_type = 6;
                    } else if (rvalue >= 85 && rvalue < 90) {
                        action_in_progress_type = 7;
                    } else if (rvalue >= 90 && rvalue < 95) {
                        action_in_progress_type = 8;
                    } else {
                        action_in_progress_type = 9;
                    }
                }
                count_of_action_in_progress = 0;
                action_in_progress = true;
                temporary_disable_cbrn = false;
                this.push_inputs_and_actions_to_buffer(inputs, action_in_progress_type);
            }
            if (action_in_progress == true) { /************ do actions from selected sequence *****************/
                switch (progressed_action[action_in_progress_type][count_of_action_in_progress]) {
                    case (action_forward): {
                        action[0] = false;
                        action[1] = true;
                        action[2] = false;
                        action[3] = false;
                        action[4] = false;
                        break;
                    }
                    case (action_forward_jump): {
                        action[0] = false;
                        action[1] = true;
                        action[2] = false;
                        action[3] = true;
                        action[4] = false;
                        break;
                    }
                    case (action_backward): {
                        action[0] = true;
                        action[1] = false;
                        action[2] = false;
                        action[3] = false;
                        action[4] = false;
                        break;
                    }
                    case (action_backward_jump): {
                        action[0] = true;
                        action[1] = false;
                        action[2] = false;
                        action[3] = true;
                        action[4] = false;
                        break;
                    }
                    case (action_forward_with_prone): {
                        action[0] = false;
                        action[1] = true;
                        action[2] = true;
                        action[3] = false;
                        action[4] = false;
                        break;
                    }
                    case (action_forward_with_jump_far): {
                        action[0] = false;
                        action[1] = true;
                        action[2] = false;
                        action[3] = true;
                        action[4] = true;
                        break;
                    }
                }
                count_of_action_in_progress++;
                if (count_of_action_in_progress >= progressed_action[action_in_progress_type].length) {
                    action_in_progress = false;
                }
            }
        } else { /************
         * selecting predefined sequence of actions when hole is near
         *****************/
            if (is_first_action_when_hole_is_detected) {
                action_in_progress_type = 2;
                count_of_action_in_progress = 0;
                is_first_action_when_hole_is_detected = false;
                this.push_inputs_and_actions_to_buffer(inputs, action_in_progress_type);
            }

            switch (progressed_action[action_in_progress_type][count_of_action_in_progress]) {
                case (action_forward): {
                    action[0] = false;
                    action[1] = true;
                    action[2] = false;
                    action[3] = false;
                    action[4] = false;
                    break;
                }
                case (action_forward_jump): {
                    action[0] = false;
                    action[1] = true;
                    action[2] = false;
                    action[3] = true;
                    action[4] = false;
                    break;
                }
                case (action_backward): {
                    action[0] = true;
                    action[1] = false;
                    action[2] = false;
                    action[3] = false;
                    action[4] = false;
                    break;
                }
                case (action_backward_jump): {
                    action[0] = true;
                    action[1] = false;
                    action[2] = false;
                    action[3] = true;
                    action[4] = false;
                    break;
                }
                case (action_forward_with_prone): {
                    action[0] = false;
                    action[1] = true;
                    action[2] = true;
                    action[3] = false;
                    action[4] = false;
                    break;
                }
                case (action_forward_with_jump_far): {
                    action[0] = false;
                    action[1] = true;
                    action[2] = false;
                    action[3] = true;
                    action[4] = true;
                    break;
                }
                case (action_none_action): {
                    action[0] = false;
                    action[1] = false;
                    action[2] = false;
                    action[3] = false;
                    action[4] = false;
                    break;
                }
            }
            count_of_action_in_progress++;
            if (action_in_progress_type == 2)
                if (count_of_action_in_progress >= progressed_action[action_in_progress_type].length) {
                    count_of_action_in_progress = 0;
                    action_in_progress_type = 0;
                    this.push_inputs_and_actions_to_buffer(inputs, action_in_progress_type);
                }

            if (action_in_progress_type == 0)
                if (count_of_action_in_progress >= progressed_action[action_in_progress_type].length) {
                    action_in_progress = false;
                    force_long_jump_forward = false;
                    is_first_action_when_hole_is_detected = true;
                }

        }
        /************ end selecting sequence of action *****************/
        push_Mario_X_pos_to_buffer(Mario_pos[0]);
        /**** Calculate mean for forwarding Mario ***/
        float forward_mean = 0;
        forward_mean = buffer_of_Mario_X[buffer_of_Mario_X.length - 1] - buffer_of_Mario_X[0];

        /********* if Mario long stay at one position ***********/
        if (forward_mean <= 0)
            if (forward_mean <= 0) {
                if (waiting_counter == 0) {
                    waiting_counter = 12;
                    action_in_progress = false;
                    temporary_disable_cbrn = true;
                } else {
                    waiting_counter--;
                }
            }
        /********* end if Mario long stay at one position ***********/

        /********* if Mario progressed well ***********/
        if (forward_mean > 0) {
            cbrn.ssetLearningRate(LearningRate);
            if ((buffer_is_full_of_Mario_X == true) && (buffer_is_full == true)) {
                for (int t = 0; t < deep_of_buffer - 1; t++) {
                    double[] targeted_outs_of_cbrn = new double[numberOfOutputs];
                    for (int i = 0; i < numberOfOutputs; i++) {
                        if (i == buffer_of_actions[t]) {
                            targeted_outs_of_cbrn[i] = 1;
                        } else {
                            targeted_outs_of_cbrn[i] = 0;
                        }
                    }
                    cbrn.propagate(buffer_of_inputs[t]);
                    cbrn.backPropagate(targeted_outs_of_cbrn);
                }
            }
        }
        /********* end if Mario progressed well ***********/

        /********* if Mario fallen to hole ***********/
        if (Mario_pos[1] > 251) {
            /********* start detector_of_holes *******/
            if (detector_of_holes_buffer_is_full) {
                double[] detector_of_holes_targeted_outs = new double[detector_of_holes_numberOfOutputs];
                // detector_of_holes.ssetLearningRate(0.1);
                detector_of_holes.ssetLearningRate(0.15);
                detector_of_holes_targeted_outs[0] = 1;
                for (int i = 0; i < 2; i++) {
                    detector_of_holes.propagate(detector_of_holes_buffer_of_inputs[i]);
                    detector_of_holes.backPropagate(detector_of_holes_targeted_outs);
                }
            }
            /********* end detector_of_holes *******/
        }
        /********* end if Mario fall to hole ***********/

        /************ if Mario don't fallen to a hole *****************/
        if (model.isMarioOnGround() && detector_of_holes_buffer_is_full) {
            /********* start detector_of_holes *******/

            double[] detector_of_holes_targeted_outs = new double[detector_of_holes_numberOfOutputs];
            detector_of_holes.ssetLearningRate(0.001);
            detector_of_holes_targeted_outs[0] = 0;
            for (int i = 0; i < 2; i++) {
                detector_of_holes.propagate(detector_of_holes_buffer_of_inputs[i]);
                detector_of_holes.backPropagate(detector_of_holes_targeted_outs);
            }
            /********* end detector_of_holes *******/
        }
        this.detector_of_holes_push_inputs_to_buffer(detector_of_holes_inputs);

        /********* if enemies hurt Mario ***********/
        if (prev_mario_mode - model.getMarioMode() > 0) {
            cbrn.ssetLearningRate(LearningRate);
            if (buffer_is_full_of_Mario_X == true) {
                double[] targeted_outs_of_cbrn = new double[numberOfOutputs];
                for (int i = 0; i < numberOfOutputs; i++) {
                    targeted_outs_of_cbrn[i] = 0;
                }
                cbrn.propagate(buffer_of_inputs[buffer_of_inputs.length - 2]);
                cbrn.backPropagate(targeted_outs_of_cbrn);
            }
        }
        /********* end if enemies hurt Mario ***********/

        prev_mario_mode = model.getMarioMode();
        return action;
    }

    @Override
    public String getAgentName() {
        return "SergeyPolikarpovAgent";
    }


}

