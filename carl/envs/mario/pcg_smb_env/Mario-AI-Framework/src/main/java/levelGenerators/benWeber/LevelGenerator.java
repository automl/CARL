package levelGenerators.benWeber;

import java.util.ArrayList;
import java.util.Random;

import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;

public class LevelGenerator implements MarioLevelGenerator {
    private int maxGaps;
    private int maxTurtles;
    private int maxCoinBlocks;

    private double CHANCE_BLOCK_POWER_UP = 0.1;
    private double CHANCE_BLOCK_COIN = 0.3;
    private double CHANCE_BLOCK_ENEMY = 0.2;
    private double CHANCE_WINGED = 0.5;
    private double CHANCE_COIN = 0.2;
    private double COIN_HEIGHT = 5;
    private double CHANCE_PLATFORM = 0.1;
    private double CHANCE_END_PLATFORM = 0.1;
    private int PLATFORM_HEIGHT = 4;
    private double CHANCE_ENEMY = 0.1;
    private double CHANCE_PIPE = 0.1;
    private int PIPE_MIN_HEIGHT = 2;
    private double PIPE_HEIGHT = 3.0;
    private int minX = 5;
    private double CHANCE_HILL = 0.1;
    private double CHANCE_END_HILL = 0.3;
    private double CHANCE_HILL_ENEMY = 0.3;
    private double HILL_HEIGHT = 4;
    private int GAP_LENGTH = 5;
    private double CHANGE_GAP = 0.1;
    private double CHANGE_HILL_CHANGE = 0.1;
    private double GAP_OFFSET = -5;
    private double GAP_RANGE = 10;
    private int GROUND_MAX_HEIGHT = 5;

    // controls the fun
    Random rand;

    // constraints
    int gapCount = 0;
    int turtleCount = 0;
    int coinBlockCount = 0;

    int xExit = 0;
    int yExit = 0;

    public LevelGenerator() {
        this(10, 7, 10);
    }

    public LevelGenerator(int maxGaps, int maxTurtles, int maxCoinBlocks) {
        this.maxGaps = maxGaps;
        this.maxTurtles = maxTurtles;
        this.maxCoinBlocks = maxCoinBlocks;
    }

    private void placeBlock(MarioLevelModel model, int x, int y) {
        // choose block type
        if (rand.nextDouble() < CHANCE_BLOCK_POWER_UP) {
            model.setBlock(x, y, MarioLevelModel.SPECIAL_BRICK);
        } else if (rand.nextDouble() < CHANCE_BLOCK_COIN && coinBlockCount < this.maxCoinBlocks) {
            model.setBlock(x, y, MarioLevelModel.COIN_BRICK);
            coinBlockCount++;
        } else {
            model.setBlock(x, y, MarioLevelModel.NORMAL_BRICK);
        }

        // place enemies
        if (rand.nextDouble() < CHANCE_BLOCK_ENEMY) {
            char t = MarioLevelModel.getEnemyCharacters(false)[this.rand.nextInt(MarioLevelModel.getEnemyCharacters(false).length)];
            // turtle constraint
            if (t == MarioLevelModel.GREEN_KOOPA || t == MarioLevelModel.RED_KOOPA) {
                if (turtleCount < this.maxTurtles) {
                    turtleCount++;
                } else {
                    t = MarioLevelModel.GOOMBA;
                }
            }
            boolean winged = rand.nextDouble() < CHANCE_WINGED;
            model.setBlock(x, y - 1, MarioLevelModel.getWingedEnemyVersion(t, winged));
        }
    }

    private void placePipe(MarioLevelModel model, int x, int y, int height) {
        model.setRectangle(x, y - height, 2, height, MarioLevelModel.PIPE);
    }

    private void setGroundHeight(MarioLevelModel model, int x, int y, int lastY, int nextY) {
        int mapHeight = model.getHeight();
        model.setRectangle(x, y + 1, 1, mapHeight - 1 - y, MarioLevelModel.GROUND);

        if (y < lastY) {
            model.setBlock(x, y, MarioLevelModel.GROUND);
            for (int i = y + 1; i <= lastY; i++) {
                model.setBlock(x, i, MarioLevelModel.GROUND);
            }
        } else if (y < nextY) {
            model.setBlock(x, y, MarioLevelModel.GROUND);
            for (int i = y + 1; i <= nextY; i++) {
                model.setBlock(x, i, MarioLevelModel.GROUND);
            }
        } else {
            model.setBlock(x, y, MarioLevelModel.GROUND);
        }

        // place the exit
        if (x == (model.getWidth() - 5)) {
            this.yExit = y - 1;
        }
    }

    public String getGeneratedLevel(MarioLevelModel model, MarioTimer timer) {
        this.rand = new Random();
        model.clearMap();

        ArrayList<Integer> ground = new ArrayList<Integer>();

        // used to place the ground
        int lastY = GROUND_MAX_HEIGHT + (int) (rand.nextDouble() * (model.getHeight() - 1 - GROUND_MAX_HEIGHT));
        int y = lastY;
        int nextY = y;
        boolean justChanged = false;
        int length = 0;
        int landHeight = model.getHeight() - 1;

        // place the ground
        for (int x = 0; x < model.getWidth(); x++) {

            // need more ground
            if (length > GAP_LENGTH && y >= model.getHeight()) {
                nextY = landHeight;
                justChanged = true;
                length = 1;
            }
            // adjust ground level
            else if (x > minX && rand.nextDouble() < CHANGE_HILL_CHANGE && !justChanged) {
                nextY += (int) (GAP_OFFSET + GAP_RANGE * rand.nextDouble());
                nextY = Math.min(model.getHeight() - 2, nextY);
                nextY = Math.max(5, nextY);
                justChanged = true;
                length = 1;
            }
            // add a gap
            else if (x > minX && y < model.getHeight() && rand.nextDouble() < CHANGE_GAP && !justChanged
                    && gapCount < this.maxGaps) {
                landHeight = Math.min(model.getHeight() - 1, lastY);
                nextY = model.getHeight();
                justChanged = true;
                length = 1;
                gapCount++;
            } else {
                length++;
                justChanged = false;
            }

            setGroundHeight(model, x, y, lastY, nextY);
            ground.add(y);

            lastY = y;
            y = nextY;
        }

        // non colliding hills
        int x = 0;
        y = model.getHeight();
        for (Integer h : ground) {
            if (y == model.getHeight()) {
                if (x > 10 && rand.nextDouble() < CHANCE_HILL) {
                    y = (int) (HILL_HEIGHT + rand.nextDouble() * (h - HILL_HEIGHT));
                    model.setBlock(x, y, MarioLevelModel.PLATFORM);
                    for (int i = y + 1; i < h; i++) {
                        model.setBlock(x, i, MarioLevelModel.PLATFORM_BACKGROUND);
                    }
                }
            } else {
                // end if hitting a wall
                if (y >= h) {
                    y = model.getHeight();
                } else if (rand.nextDouble() < CHANCE_END_HILL) {
                    model.setBlock(x, y, MarioLevelModel.PLATFORM);
                    for (int i = y + 1; i < h; i++) {
                        model.setBlock(x, i, MarioLevelModel.PLATFORM_BACKGROUND);
                    }

                    y = model.getHeight();
                } else {
                    model.setBlock(x, y, MarioLevelModel.PLATFORM);
                    for (int i = y + 1; i < h; i++) {
                        model.setBlock(x, i, MarioLevelModel.PLATFORM_BACKGROUND);
                    }

                    if (rand.nextDouble() < CHANCE_HILL_ENEMY) {
                        char t = MarioLevelModel.getEnemyCharacters(false)[this.rand.nextInt(MarioLevelModel.getEnemyCharacters(false).length)];
                        // turtle constraint
                        if (t == MarioLevelModel.GREEN_KOOPA || t == MarioLevelModel.RED_KOOPA) {
                            if (turtleCount < this.maxTurtles) {
                                turtleCount++;
                            } else {
                                t = MarioLevelModel.GOOMBA;
                            }
                        }
                        boolean winged = rand.nextDouble() < CHANCE_WINGED;
                        model.setBlock(x, y - 1, MarioLevelModel.getWingedEnemyVersion(t, winged));
                    }
                }
            }

            x++;
        }

        // pipes
        lastY = 0;
        int lastlastY = 0;
        x = 0;
        int lastX = 0;
        for (Integer h : ground) {
            if (x > minX && rand.nextDouble() < CHANCE_PIPE) {
                if (h == lastY && lastlastY <= lastY && x > (lastX + 1)) {
                    int height = PIPE_MIN_HEIGHT + (int) (Math.random() * PIPE_HEIGHT);
                    placePipe(model, x - 1, h, height);
                    lastX = x;
                }
            }

            lastlastY = lastY;
            lastY = h;
            x++;
        }

        // place enemies
        x = 0;
        for (Integer h : ground) {
            if (x > minX && rand.nextDouble() < CHANCE_ENEMY) {
                char t = MarioLevelModel.getEnemyCharacters(false)[this.rand.nextInt(MarioLevelModel.getEnemyCharacters(false).length)];
                // turtle constraint
                if (t == MarioLevelModel.GREEN_KOOPA || t == MarioLevelModel.RED_KOOPA) {
                    if (turtleCount < this.maxTurtles) {
                        turtleCount++;
                    } else {
                        t = MarioLevelModel.GOOMBA;
                    }
                }
                boolean winged = rand.nextDouble() < CHANCE_WINGED;
                char tile = model.getBlock(x, h - 1);
                if (tile == MarioLevelModel.EMPTY) {
                    model.setBlock(x, h - 1, MarioLevelModel.getWingedEnemyVersion(t, winged));
                }
            }
            x++;
        }

        // platforms
        x = 0;
        y = model.getHeight();
        for (Integer h : ground) {
            int max = 0;

            // find the highest object
            for (max = 0; max < h; max++) {
                int tile = model.getBlock(x, max);
                if (tile != 0) {
                    break;
                }
            }

            if (y == model.getHeight()) {
                if (x > minX && rand.nextDouble() < CHANCE_PLATFORM) {
                    y = max - PLATFORM_HEIGHT; // (int)(-5*rand.nextDouble()*(h - 0));

                    if (y >= 1 && h - max > 1) {
                        placeBlock(model, x, y);
                    } else {
                        y = model.getHeight();
                    }
                }
            } else {
                // end if hitting a wall
                if (y >= (max + 1)) {
                    y = model.getHeight();
                } else if (rand.nextDouble() < CHANCE_END_PLATFORM) {
                    placeBlock(model, x, y);
                    y = model.getHeight();
                } else {
                    placeBlock(model, x, y);
                }
            }
            x++;
        }

        // coins
        x = 0;
        for (Integer h : ground) {
            if (x > 5 && rand.nextDouble() < CHANCE_COIN) {
                y = h - (int) (1 + Math.random() * COIN_HEIGHT);

                char tile = model.getBlock(x, y);
                if (tile == MarioLevelModel.EMPTY) {
                    model.setBlock(x, y, MarioLevelModel.COIN);
                }
            }

            x++;
        }
        // place the exit
        this.xExit = model.getWidth() - 5;
        model.setBlock(this.xExit, this.yExit, MarioLevelModel.MARIO_EXIT);
        return model.getMap();
    }

    public String getGeneratorName() {
        return "BenWeberLevelGenerator";
    }
}
