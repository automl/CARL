package levelGenerators.random;

import java.util.Random;

import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;

public class LevelGenerator implements MarioLevelGenerator {
    private final int GROUND_Y_LOCATION = 13;
    private final float GROUND_PROB = 0.4f;
    private final int OBSTACLES_LOCATION = 10;
    private final float OBSTACLES_PROB = 0.1f;
    private final int COLLECTIBLE_LOCATION = 3;
    private final float COLLECTIBLE_PROB = 0.05f;
    private final float ENMEY_PROB = 0.1f;
    private final int FLOOR_PADDING = 3;

    @Override
    public String getGeneratedLevel(MarioLevelModel model, MarioTimer timer) {
        Random random = new Random();
        model.clearMap();
        for (int x = 0; x < model.getWidth(); x++) {
            for (int y = 0; y < model.getHeight(); y++) {
                model.setBlock(x, y, MarioLevelModel.EMPTY);
                if (y > GROUND_Y_LOCATION) {
                    if (random.nextDouble() < GROUND_PROB) {
                        model.setBlock(x, y, MarioLevelModel.GROUND);
                    }
                } else if (y > OBSTACLES_LOCATION) {
                    if (random.nextDouble() < OBSTACLES_PROB) {
                        model.setBlock(x, y, MarioLevelModel.PYRAMID_BLOCK);
                    } else if (random.nextDouble() < ENMEY_PROB) {
                        model.setBlock(x, y,
                                MarioLevelModel.getEnemyCharacters()[random.nextInt(MarioLevelModel.getEnemyCharacters().length)]);
                    }
                } else if (y > COLLECTIBLE_LOCATION) {
                    if (random.nextDouble() < COLLECTIBLE_PROB) {
                        model.setBlock(x, y,
                                MarioLevelModel.getCollectablesTiles()[random.nextInt(MarioLevelModel.getCollectablesTiles().length)]);
                    }
                }
            }
        }
        model.setRectangle(0, 14, FLOOR_PADDING, 2, MarioLevelModel.GROUND);
        model.setRectangle(model.getWidth() - 1 - FLOOR_PADDING, 14, FLOOR_PADDING, 2, MarioLevelModel.GROUND);
        model.setBlock(FLOOR_PADDING / 2, 13, MarioLevelModel.MARIO_START);
        model.setBlock(model.getWidth() - 1 - FLOOR_PADDING / 2, 13, MarioLevelModel.MARIO_EXIT);
        return model.getMap();
    }

    @Override
    public String getGeneratorName() {
        return "RandomLevelGenerator";
    }

}
