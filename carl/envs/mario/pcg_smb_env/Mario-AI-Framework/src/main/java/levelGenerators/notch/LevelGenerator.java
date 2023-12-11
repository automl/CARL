package levelGenerators.notch;

import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;

import java.util.Random;

public class LevelGenerator implements MarioLevelGenerator {
    private static final int ODDS_STRAIGHT = 0;
    private static final int ODDS_HILL_STRAIGHT = 1;
    private static final int ODDS_TUBES = 2;
    private static final int ODDS_JUMP = 3;
    private static final int ODDS_CANNONS = 4;

    private int[] odds = new int[5];
    private int totalOdds;
    private int difficulty;
    private int type;
    private Random random;

    public LevelGenerator() {
        random = new Random();
        this.type = random.nextInt(3);
        this.difficulty = random.nextInt(5);
    }

    public LevelGenerator(int type, int difficulty) {
        random = new Random();
        this.type = type;
        this.difficulty = difficulty;
    }

    private int buildZone(MarioLevelModel model, int x, int maxLength) {
        int t = random.nextInt(totalOdds);
        int type = 0;
        for (int i = 0; i < odds.length; i++) {
            if (odds[i] <= t) {
                type = i;
            }
        }

        switch (type) {
            case ODDS_STRAIGHT:
                return buildStraight(model, x, maxLength, false);
            case ODDS_HILL_STRAIGHT:
                return buildHillStraight(model, x, maxLength);
            case ODDS_TUBES:
                return buildTubes(model, x, maxLength);
            case ODDS_JUMP:
                return buildJump(model, x, maxLength);
            case ODDS_CANNONS:
                return buildCannons(model, x, maxLength);
        }
        return 0;
    }

    private int buildJump(MarioLevelModel model, int xo, int maxLength) {
        int js = random.nextInt(4) + 2;
        int jl = random.nextInt(2) + 2;
        int length = js * 2 + jl;

        boolean hasStairs = random.nextInt(3) == 0;

        int floor = model.getHeight() - 1 - random.nextInt(4);
        for (int x = xo; x < xo + length; x++) {
            if (x < xo + js || x > xo + length - js - 1) {
                for (int y = 0; y < model.getHeight(); y++) {
                    if (y >= floor) {
                        model.setBlock(x, y, MarioLevelModel.GROUND);
                    } else if (hasStairs) {
                        if (x < xo + js) {
                            if (y >= floor - (x - xo) + 1) {
                                model.setBlock(x, y, MarioLevelModel.GROUND);
                            }
                        } else {
                            if (y >= floor - ((xo + length) - x) + 2) {
                                model.setBlock(x, y, MarioLevelModel.GROUND);
                            }
                        }
                    }
                }
            }
        }

        return length;
    }

    private int buildCannons(MarioLevelModel model, int xo, int maxLength) {
        int length = random.nextInt(10) + 2;
        if (length > maxLength)
            length = maxLength;

        int floor = model.getHeight() - 1 - random.nextInt(4);
        int xCannon = xo + 1 + random.nextInt(4);
        for (int x = xo; x < xo + length; x++) {
            if (x > xCannon) {
                xCannon += 2 + random.nextInt(4);
            }
            if (xCannon == xo + length - 1)
                xCannon += 10;
            int cannonHeight = floor - random.nextInt(4) - 1;

            for (int y = 0; y < model.getHeight(); y++) {
                if (y >= floor) {
                    model.setBlock(x, y, MarioLevelModel.GROUND);
                } else {
                    if (x == xCannon && y >= cannonHeight) {
                        model.setBlock(x, y, MarioLevelModel.BULLET_BILL);
                    }
                }
            }
        }

        return length;
    }

    private int buildHillStraight(MarioLevelModel model, int xo, int maxLength) {
        int length = random.nextInt(10) + 10;
        if (length > maxLength)
            length = maxLength;

        int floor = model.getHeight() - 1 - random.nextInt(4);
        for (int x = xo; x < xo + length; x++) {
            for (int y = 0; y < model.getHeight(); y++) {
                if (y >= floor) {
                    model.setBlock(x, y, MarioLevelModel.GROUND);
                }
            }
        }

        addEnemyLine(model, xo + 1, xo + length - 1, floor - 1);

        int h = floor;

        boolean keepGoing = true;

        boolean[] occupied = new boolean[length];
        while (keepGoing) {
            h = h - 2 - random.nextInt(3);

            if (h <= 0) {
                keepGoing = false;
            } else {
                int l = random.nextInt(5) + 3;
                int xxo = random.nextInt(length - l - 2) + xo + 1;

                if (occupied[xxo - xo] || occupied[xxo - xo + l] || occupied[xxo - xo - 1]
                        || occupied[xxo - xo + l + 1]) {
                    keepGoing = false;
                } else {
                    occupied[xxo - xo] = true;
                    occupied[xxo - xo + l] = true;
                    addEnemyLine(model, xxo, xxo + l, h - 1);
                    if (random.nextInt(4) == 0) {
                        decorate(model, xxo - 1, xxo + l + 1, h);
                        keepGoing = false;
                    }
                    for (int x = xxo; x < xxo + l; x++) {
                        for (int y = h; y < floor; y++) {
                            int yy = 9;
                            if (y == h)
                                yy = 8;
                            if (model.getBlock(x, y) == MarioLevelModel.EMPTY) {
                                if (yy == 8) {
                                    model.setBlock(x, y, MarioLevelModel.PLATFORM);
                                } else {
                                    model.setBlock(x, y, MarioLevelModel.PLATFORM_BACKGROUND);
                                }
                            }
                        }
                    }
                }
            }
        }

        return length;
    }

    private void addEnemyLine(MarioLevelModel model, int x0, int x1, int y) {
        char[] enemies = new char[]{MarioLevelModel.GOOMBA,
                MarioLevelModel.GREEN_KOOPA,
                MarioLevelModel.RED_KOOPA,
                MarioLevelModel.SPIKY};
        for (int x = x0; x < x1; x++) {
            if (random.nextInt(35) < difficulty + 1) {
                int type = random.nextInt(4);
                if (difficulty < 1) {
                    type = 0;
                } else if (difficulty < 3) {
                    type = 1 + random.nextInt(3);
                }
                model.setBlock(x, y, MarioLevelModel.getWingedEnemyVersion(enemies[type], random.nextInt(35) < difficulty));
            }
        }
    }

    private int buildTubes(MarioLevelModel model, int xo, int maxLength) {
        int length = random.nextInt(10) + 5;
        if (length > maxLength)
            length = maxLength;

        int floor = model.getHeight() - 1 - random.nextInt(4);
        int xTube = xo + 1 + random.nextInt(4);
        int tubeHeight = floor - random.nextInt(2) - 2;
        for (int x = xo; x < xo + length; x++) {
            if (x > xTube + 1) {
                xTube += 3 + random.nextInt(4);
                tubeHeight = floor - random.nextInt(2) - 2;
            }
            if (xTube >= xo + length - 2)
                xTube += 10;

            char tubeType = MarioLevelModel.PIPE;
            if (x == xTube && random.nextInt(11) < difficulty + 1) {
                tubeType = MarioLevelModel.PIPE_FLOWER;
            }

            for (int y = 0; y < model.getHeight(); y++) {
                if (y >= floor) {
                    model.setBlock(x, y, MarioLevelModel.GROUND);
                } else {
                    if ((x == xTube || x == xTube + 1) && y >= tubeHeight) {
                        model.setBlock(x, y, tubeType);
                    }
                }
            }
        }

        return length;
    }

    private int buildStraight(MarioLevelModel model, int xo, int maxLength, boolean safe) {
        int length = random.nextInt(10) + 2;
        if (safe)
            length = 10 + random.nextInt(5);
        if (length > maxLength)
            length = maxLength;

        int floor = model.getHeight() - 1 - random.nextInt(4);
        for (int x = xo; x < xo + length; x++) {
            for (int y = 0; y < model.getHeight(); y++) {
                if (y >= floor) {
                    model.setBlock(x, y, MarioLevelModel.GROUND);
                }
            }
        }

        if (!safe) {
            if (length > 5) {
                decorate(model, xo, xo + length, floor);
            }
        }

        return length;
    }

    private void decorate(MarioLevelModel model, int x0, int x1, int floor) {
        if (floor < 1)
            return;

        boolean rocks = true;
        addEnemyLine(model, x0 + 1, x1 - 1, floor - 1);

        int s = random.nextInt(4);
        int e = random.nextInt(4);

        if (floor - 2 > 0) {
            if ((x1 - 1 - e) - (x0 + 1 + s) > 1) {
                for (int x = x0 + 1 + s; x < x1 - 1 - e; x++) {
                    model.setBlock(x, floor - 2, MarioLevelModel.COIN);
                }
            }
        }

        s = random.nextInt(4);
        e = random.nextInt(4);

        if (floor - 4 > 0) {
            if ((x1 - 1 - e) - (x0 + 1 + s) > 2) {
                for (int x = x0 + 1 + s; x < x1 - 1 - e; x++) {
                    if (rocks) {
                        if (x != x0 + 1 && x != x1 - 2 && random.nextInt(3) == 0) {
                            if (random.nextInt(4) == 0) {
                                model.setBlock(x, floor - 4, MarioLevelModel.NORMAL_BRICK);
                            } else {
                                model.setBlock(x, floor - 4, MarioLevelModel.NORMAL_BRICK);
                            }
                        } else if (random.nextInt(4) == 0) {
                            if (random.nextInt(4) == 0) {
                                model.setBlock(x, floor - 4, MarioLevelModel.COIN);
                            } else {
                                model.setBlock(x, floor - 4, MarioLevelModel.COIN);
                            }
                        } else {
                            model.setBlock(x, floor - 4, MarioLevelModel.COIN);
                        }
                    }
                }
            }
        }
    }

    @Override
    public String getGeneratedLevel(MarioLevelModel model, MarioTimer timer) {
        model.clearMap();

        odds[ODDS_STRAIGHT] = 20;
        odds[ODDS_HILL_STRAIGHT] = 10;
        odds[ODDS_TUBES] = 2 + 1 * difficulty;
        odds[ODDS_JUMP] = 2 * difficulty;
        odds[ODDS_CANNONS] = -10 + 5 * difficulty;

        if (type > 0) {
            odds[ODDS_HILL_STRAIGHT] = 0;
        }

        for (int i = 0; i < odds.length; i++) {
            if (odds[i] < 0)
                odds[i] = 0;
            totalOdds += odds[i];
            odds[i] = totalOdds - odds[i];
        }

        int length = 0;
        length += buildStraight(model, 0, model.getWidth(), true);
        while (length < model.getWidth()) {
            length += buildZone(model, length, model.getWidth() - length);
        }

        int floor = model.getHeight() - 1 - random.nextInt(4);

        for (int x = length; x < model.getWidth(); x++) {
            for (int y = 0; y < model.getHeight(); y++) {
                if (y >= floor) {
                    model.setBlock(x, y, MarioLevelModel.GROUND);
                }
            }
        }

        if (type > 0) {
            int ceiling = 0;
            int run = 0;
            for (int x = 0; x < model.getWidth(); x++) {
                if (run-- <= 0 && x > 4) {
                    ceiling = random.nextInt(4);
                    run = random.nextInt(4) + 4;
                }
                for (int y = 0; y < model.getHeight(); y++) {
                    if ((x > 4 && y <= ceiling) || x < 1) {
                        model.setBlock(x, y, MarioLevelModel.NORMAL_BRICK);
                    }
                }
            }
        }
        return model.getMap();
    }

    @Override
    public String getGeneratorName() {
        return "NotchLevelGenerator";
    }
}