package agents.andySloane;

import engine.core.MarioForwardModel;
import engine.core.MarioGame;

public class Sensors {
    private String[][] asciiScene;
    public int[][] levelScene;
    public int[][] enemiesScene;
    public int fireballsOnScreen;

    public void updateReadings(MarioForwardModel model) {
        levelScene = model.getMarioSceneObservation();
        enemiesScene = model.getMarioEnemiesObservation();

        asciiScene = new String[MarioGame.tileWidth][MarioGame.tileHeight];

        fireballsOnScreen = 0;
        for (int x = 0; x < levelScene.length; ++x)
            for (int y = 0; y < levelScene[0].length; ++y)
                asciiScene[x][y] = asciiLevel(levelScene[x][y]);
        for (int x = 0; x < enemiesScene.length; ++x)
            for (int y = 0; y < enemiesScene[0].length; ++y) {
                int enemy = enemiesScene[x][y];
                if (enemy == MarioForwardModel.OBS_NONE)
                    continue;
                if (enemy == MarioForwardModel.OBS_FIREBALL)
                    fireballsOnScreen++;
                asciiScene[x][y] = asciiEnemy(enemy);
            }
    }

    public int[] getMarioPosition() {
        return new int[]{8, 8};
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (String[] sceneRow : asciiScene) {
            for (String square : sceneRow)
                sb.append(square + " ");
            sb.append('\n');
        }
        return sb.toString();
    }

    public final static int EMPTY = 0;
    public final static int COIN = 31;
    public final static int SOLID = 17;
    public final static int PLATFORM = 59;
    public final static int QUESTIONMARK_BOX = 24;
    public final static int BRICK = 23;

    private String asciiLevel(int levelSquare) {
        switch (levelSquare) {
            case EMPTY:
                return " ";
            case COIN:
                return "O";
            case SOLID:
                return "X";
            case PLATFORM:
                return "-";
            case BRICK:
                return "B";
            case QUESTIONMARK_BOX:
                return "?";
            default:
                return "" + levelSquare;
        }
    }

    private String asciiEnemy(int enemySquare) {
        if (enemySquare == MarioForwardModel.OBS_GOOMBA) {
            return "G";
        }
        if (enemySquare == MarioForwardModel.OBS_RED_KOOPA || enemySquare == MarioForwardModel.OBS_GREEN_KOOPA) {
            return "n";
        }
        if (enemySquare == MarioForwardModel.OBS_GOOMBA_WINGED || enemySquare == MarioForwardModel.OBS_RED_KOOPA_WINGED ||
                enemySquare == MarioForwardModel.OBS_GREEN_KOOPA_WINGED) {
            return "w";
        }
        if (enemySquare == MarioForwardModel.OBS_SHELL) {
            return "D";
        }
        if (enemySquare == MarioForwardModel.OBS_SPIKY) {
            return "^";
        }
        if (enemySquare == MarioForwardModel.OBS_SPIKY_WINGED) {
            return "W";
        }
        if (enemySquare == MarioForwardModel.OBS_BULLET_BILL) {
            return "<";
        }
        if (enemySquare == MarioForwardModel.OBS_ENEMY_FLOWER) {
            return "V";
        }
        if (enemySquare == MarioForwardModel.OBS_FIRE_FLOWER) {
            return "F";
        }
        if (enemySquare == MarioForwardModel.OBS_FIREBALL) {
            return "*";
        }
        return "" + enemySquare;
    }
}
