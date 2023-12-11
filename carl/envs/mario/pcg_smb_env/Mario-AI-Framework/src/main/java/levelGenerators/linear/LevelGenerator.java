package levelGenerators.linear;

import java.util.ArrayList;
import java.util.Random;

import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;

public class LevelGenerator implements MarioLevelGenerator {
    private final int GROUND_PADDING = 5;
    private final int GROUND_LENGTH = 8;
    private final int GAP_LENGTH = 6;
    private final float GAP_PROB = 0.1f;
    private final float PIPE_PROB = 0.75f;
    private final int GROUND_PIPE_LENGTH = 10;
    private final int PIPE_HEIGHT = 6;
    private final float COLLECTIBLE_PROB = 0.75f;
    private final int GROUND_COLLECTIBLE_LENGTH = 6;
    private final int GROUND_ENEMY_LENGTH = 2;

    private Random rnd;

    private void placePipe(MarioLevelModel model, int x, int y, int height) {
        char pipeType = MarioLevelModel.PIPE;
        if (this.rnd.nextDouble() < 0.2) {
            pipeType = MarioLevelModel.PIPE_FLOWER;
        }
        model.setRectangle(x, y - height + 1, 2, height, pipeType);
    }

    private void placeInterestingArrangement(MarioLevelModel model, int x, int y, int width) {
        for (int i = 0; i < width / 2; i++) {
            char type = MarioLevelModel.getBumpableTiles()[this.rnd.nextInt(MarioLevelModel.getBumpableTiles().length)];
            model.setBlock(x + i, y, type);
            model.setBlock(x + width - 1 - i, y, type);
        }

        if (width % 2 == 1 && this.rnd.nextDouble() < 0.25) {
            char type = MarioLevelModel.getBumpableTiles()[this.rnd.nextInt(MarioLevelModel.getBumpableTiles().length)];
            model.setBlock(x + width / 2, y, type);
        }

        if (y > 4 && this.rnd.nextDouble() < 0.25) {
            this.placeInterestingArrangement(model, x + width / 4, y - 3 - this.rnd.nextInt(3), width / 2);
        }
    }

    private void placeEnemy(MarioLevelModel model, int x1, int x2, int y) {
        boolean winged = this.rnd.nextDouble() < 0.1;
        char enemyType = MarioLevelModel.getEnemyCharacters(false)[this.rnd.nextInt(MarioLevelModel.getEnemyCharacters(false).length)];
        enemyType = MarioLevelModel.getWingedEnemyVersion(enemyType, winged);
        int xStart = x1 + this.rnd.nextInt(x2 - x1);
        int length = 1 + this.rnd.nextInt(3);
        if (length > x2 - x1 - 1) {
            length = x2 - x1 - 1;
        }
        for (int i = 0; i < length; i++) {
            if (model.getBlock(xStart + i, y) == MarioLevelModel.EMPTY)
                model.setBlock(xStart + i, y, enemyType);
        }
    }

    @Override
    public String getGeneratedLevel(MarioLevelModel model, MarioTimer timer) {
        this.rnd = new Random();
        model.clearMap();

        ArrayList<Integer> groundArea = new ArrayList<Integer>();
        groundArea.add(0);
        int groundLength = GROUND_LENGTH / 2 + this.rnd.nextInt(GROUND_LENGTH / 2);
        int gapLength = 0;

        //add ground
        for (int x = 0; x < model.getWidth(); x++) {
            if (groundLength > 0 || gapLength == 0 || x < GROUND_PADDING || x > model.getWidth() - 1 - GROUND_PADDING) {
                model.setBlock(x, model.getHeight() - 1, MarioLevelModel.GROUND);
                model.setBlock(x, model.getHeight() - 2, MarioLevelModel.GROUND);
                groundLength -= 1;
                if (groundLength <= 0 && this.rnd.nextDouble() < GAP_PROB) {
                    gapLength = GAP_LENGTH / 2 + this.rnd.nextInt(GAP_LENGTH / 2);
                }
                if (groundArea.size() % 2 == 0) {
                    groundArea.add(x);
                }
            } else {
                gapLength -= 1;
                if (gapLength <= 0) {
                    groundLength = GROUND_LENGTH / 2 + this.rnd.nextInt(GROUND_LENGTH / 2);
                }
                if (groundArea.size() % 2 == 1) {
                    groundArea.add(x);
                }
            }
        }
        groundArea.add(model.getWidth() - 1);

        //add pipes
        ArrayList<Integer> newAreas = new ArrayList<Integer>();
        for (int i = 0; i < groundArea.size() / 2; i++) {
            groundLength = groundArea.get(2 * i + 1) - groundArea.get(2 * i);
            if (groundLength > GROUND_PIPE_LENGTH && this.rnd.nextDouble() < PIPE_PROB) {
                int x = groundArea.get(2 * i) + this.rnd.nextInt(groundLength / 4) + 3;
                this.placePipe(model, x, model.getHeight() - 3, this.rnd.nextInt(PIPE_HEIGHT * 2 / 3) + PIPE_HEIGHT / 3);
                newAreas.add(groundArea.get(2 * i));
                newAreas.add(x - 1);
                newAreas.add(x + 2);
                newAreas.add(groundArea.get(2 * i + 1));
            }
        }

        //add interesting patterns
        groundArea.clear();
        for (int i = 0; i < newAreas.size() / 2; i++) {
            groundLength = newAreas.get(2 * i + 1) - newAreas.get(2 * i);
            groundArea.add(newAreas.get(2 * i));
            groundArea.add(model.getHeight() - 3);
            groundArea.add(newAreas.get(2 * i + 1));
            groundArea.add(model.getHeight() - 3);
            if (groundLength > GROUND_COLLECTIBLE_LENGTH && this.rnd.nextDouble() < COLLECTIBLE_PROB) {
                int x = newAreas.get(2 * i) + this.rnd.nextInt(groundLength / 3) + 1;
                int y = model.getHeight() - 5 - this.rnd.nextInt(3);
                int length = 1 + this.rnd.nextInt(groundLength / 3);
                this.placeInterestingArrangement(model, x, y, length);
                groundArea.add(x + 1);
                groundArea.add(y - 1);
                groundArea.add(x + length - 1);
                groundArea.add(y - 1);
            }
        }

        //add enemies
        for (int i = 1; i < groundArea.size() / 4; i++) {
            groundLength = groundArea.get(4 * i + 2) - groundArea.get(4 * i);
            if (groundLength > GROUND_ENEMY_LENGTH) {
                this.placeEnemy(model, groundArea.get(4 * i), groundArea.get(4 * i + 2), groundArea.get(4 * i + 1));
            }
        }

        model.setBlock(1, model.getHeight() - 3, MarioLevelModel.MARIO_START);
        model.setBlock(model.getWidth() - 2, model.getHeight() - 3, MarioLevelModel.PYRAMID_BLOCK);
        model.setBlock(model.getWidth() - 2, model.getHeight() - 4, MarioLevelModel.MARIO_EXIT);
        return model.getMap();
    }

    @Override
    public String getGeneratorName() {
        return "LinearLevelGenerator";
    }

}
