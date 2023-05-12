package agents.michal;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.MarioActions;

public class Agent implements MarioAgent {
    private enum STATE {
        WALK_FORWARD, WALK_BACKWARD, JUMP, JUMP_HOLE
    }

    private boolean facing_left;
    private int leftCounter;
    private int shootCounter;
    private STATE state;
    private boolean[] action;

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        action = new boolean[MarioActions.numberOfActions()];
        state = STATE.WALK_FORWARD;
        facing_left = false;

        leftCounter = 0;
        shootCounter = 0;
    }

    private int getLocation(int relX, int relY, int[][] scene) {
        int realX = 8 + relX;
        int realY = 8 + relY;

        return scene[realX][realY];
    }

    private boolean thereIsObstacle(int[][] scene) {
        int[] inFrontOf = new int[]{getLocation(1, 0, scene), getLocation(2, 0, scene), getLocation(2, -1, scene)};

        for (int i = 0; i < inFrontOf.length; i++) {
            if (inFrontOf[i] == 17 || inFrontOf[i] == 23 || inFrontOf[i] == 24) {
                return true;
            }
        }

        return false;
    }

    private boolean thereIsHole(int[][] scene) {
        for (int i = 1; i < 3; i++) {
            for (int j = 2; j < 8; j++) {
                if (getLocation(i, j, scene) != 0) {
                    return false;
                }
            }
        }

        return true;
    }

    private boolean enemyInFront(int[][] enemies) {
        for (int i = 0; i > -2; i--) {
            for (int j = 1; j < 2; j++) {
                if (getLocation(j, i, enemies) > 1) {
                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        int[][] scene = model.getMarioSceneObservation();
        int[][] enemies = model.getMarioEnemiesObservation();

        if (enemyInFront(enemies)) {
            if (shootCounter > 0) {
                action[MarioActions.SPEED.getValue()] = false;
            } else {
                action[MarioActions.SPEED.getValue()] = true;
                shootCounter++;
            }
            return action;
        } else if (shootCounter > 0) {
            shootCounter = 0;
        }

        switch (state) {
            case WALK_BACKWARD:
                if (leftCounter > 5) {
                    state = STATE.WALK_FORWARD;
                    facing_left = false;
                }
                leftCounter++;
                action[MarioActions.LEFT.getValue()] = true;
                action[MarioActions.RIGHT.getValue()] = false;

                break;

            case WALK_FORWARD:
                action[MarioActions.LEFT.getValue()] = false;
                if (thereIsHole(scene)) {
                    state = STATE.JUMP_HOLE;
                    action[MarioActions.JUMP.getValue()] = true;
                    action[MarioActions.SPEED.getValue()] = true;
                } else if (thereIsObstacle(scene)) {
                    state = STATE.JUMP;
                    action[MarioActions.JUMP.getValue()] = true;
                    action[MarioActions.RIGHT.getValue()] = true;
                    action[MarioActions.SPEED.getValue()] = false;
                } else {
                    action[MarioActions.RIGHT.getValue()] = true;
                    action[MarioActions.SPEED.getValue()] = false;
                }
                break;

            case JUMP:
                if (action[MarioActions.RIGHT.getValue()] && thereIsHole(scene)) {
                    action[MarioActions.RIGHT.getValue()] = false;
                    action[MarioActions.LEFT.getValue()] = true;

                    facing_left = true;
                } else if (model.isMarioOnGround()) {
                    if (facing_left) {
                        state = STATE.WALK_BACKWARD;
                        leftCounter = 0;
                    } else {
                        state = STATE.WALK_FORWARD;
                    }

                    action[MarioActions.LEFT.getValue()] = false;
                    action[MarioActions.RIGHT.getValue()] = false;

                    action[MarioActions.JUMP.getValue()] = false;
                    action[MarioActions.SPEED.getValue()] = false;
                }
                break;

            case JUMP_HOLE:
                if (model.isMarioOnGround()) {
                    state = STATE.WALK_FORWARD;

                    action[MarioActions.JUMP.getValue()] = false;
                    action[MarioActions.SPEED.getValue()] = false;

                    action[MarioActions.LEFT.getValue()] = false;
                    action[MarioActions.RIGHT.getValue()] = false;
                }
                break;
        }

        return action;
    }

    @Override
    public String getAgentName() {
        return "MichalAgent";
    }
}
