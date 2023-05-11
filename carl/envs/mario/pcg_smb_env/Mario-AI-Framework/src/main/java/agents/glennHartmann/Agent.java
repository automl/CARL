package agents.glennHartmann;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.MarioActions;

public class Agent implements MarioAgent {
    private boolean[] action = new boolean[MarioActions.numberOfActions()];
    private int jumpCount = 0; // counter to determine if you've done a 'full' jump yet
    private int speedCount = 0; // counter to determine if you should shoot again

    // determines if jumping will cause you to hit an enemy or not
    private boolean safeToJumpFromEnemies(byte[][] enemiesFromBitmap) {
        for (int y = 5; y <= 9; y++) {
            for (int x = 11; x <= 14; x++) {
                if (!(x == 8 && y == 8) && enemiesFromBitmap[x][y] == 1) {
                    return false;
                }
            }
        }

        return true;
    }

    // determines if jumping will land you in a gap
    private boolean safeToJumpFromGaps(byte[][] levelSceneFromBitmap) {
        for (int y = 9; y <= 9; y++) {
            boolean b = false;
            for (int x = 11; x <= 14; x++) {
                if (levelSceneFromBitmap[x][y] == 1) {
                    b = true;
                    break;
                }
            }
            if (!b) {
                return false;
            }
        }

        return true;
    }

    // determines if there are enemies close enough to pose a danger to you -
    // implies you should jump
    private boolean dangerFromEnemies(byte[][] enemiesFromBitmap) {
        for (int y = 7; y <= 9; y++) {
            for (int x = 8; x <= 12; x++) {
                if (!(x == 8 && y == 8) && enemiesFromBitmap[x][y] == 1) {
                    return true;
                }
            }
        }

        return false;
    }

    // determines if there is a gap close enough to pose a danger to you - implies
    // you should jump
    private boolean dangerFromGaps(byte[][] levelSceneFromBitmap) {
        for (int y = 9; y <= 10; y++) {
            for (int x = 9; x <= 12; x++) {
                if (levelSceneFromBitmap[x][y] == 0) {
                    return true;
                }
            }
        }

        return false;
    }

    // determines if it's safe to jump
    private boolean safeToJump(byte[][] levelSceneFromBitmap, byte[][] enemiesFromBitmap) {
        return safeToJumpFromGaps(levelSceneFromBitmap) && safeToJumpFromEnemies(enemiesFromBitmap);
    }

    // determines if there is something blocking your path that you need to jump
    // over
    private boolean block(byte[][] levelSceneFromBitmap) {
        for (int y = 8; y <= 8; y++) {
            for (int x = 9; x <= 12; x++) {
                if (levelSceneFromBitmap[x][y] == 1) {
                    return true;
                }
            }
        }

        return false;
    }

    // function from ForwardAgent.java - I did not write this
    private byte[][] decode(MarioForwardModel model, int[][] state) {
        byte[][] dstate = new byte[model.obsGridWidth][model.obsGridHeight];
        for (int i = 0; i < dstate.length; ++i)
            for (int j = 0; j < dstate[0].length; ++j)
                dstate[i][j] = 2;

        for (int x = 0; x < state.length; x++) {
            for (int y = 0; y < state[x].length; y++) {
                if (state[x][y] != 0) {
                    dstate[x][y] = 1;
                } else {
                    dstate[x][y] = 0;
                }
            }
        }
        return dstate;
    }

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        action = new boolean[MarioActions.numberOfActions()];
        action[MarioActions.RIGHT.getValue()] = true;
        action[MarioActions.SPEED.getValue()] = true;
        action[MarioActions.JUMP.getValue()] = false;
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        byte[][] levelSceneFromBitmap = decode(model, model.getMarioSceneObservation()); // map of the scene
        byte[][] enemiesFromBitmap = decode(model, model.getMarioEnemiesObservation()); // map of enemies

        // if jump is active and jumpCount is too big, deactivate - jump is over and
        // you'll need to get ready for next one
        if (action[MarioActions.JUMP.getValue()] && jumpCount >= 8) {
            action[MarioActions.JUMP.getValue()] = false;
            jumpCount = 0;
        }
        // otherwise you're in the middle of jump, increment jumpCount
        else if (action[MarioActions.JUMP.getValue()]) {
            jumpCount++;
        }
        // now, if you're in danger from enemies, or blocked by landscape, jump if it's
        // safe to. If there's danger of falling, jump no matter what
        else if ((((dangerFromEnemies(enemiesFromBitmap) || block(levelSceneFromBitmap))
                && safeToJump(levelSceneFromBitmap, enemiesFromBitmap)) || dangerFromGaps(levelSceneFromBitmap))
                && model.mayMarioJump()) {
            action[MarioActions.JUMP.getValue()] = true;
        }

        // keep shooting
        if (action[MarioActions.SPEED.getValue()] && speedCount >= 10) {
            action[MarioActions.SPEED.getValue()] = false;
            speedCount = 0;
        } else if (action[MarioActions.SPEED.getValue()]) {
            speedCount++;
        } else {
            action[MarioActions.SPEED.getValue()] = true;
        }

        return action;
    }

    @Override
    public String getAgentName() {
        return "GlennHartmannAgent";
    }
}
