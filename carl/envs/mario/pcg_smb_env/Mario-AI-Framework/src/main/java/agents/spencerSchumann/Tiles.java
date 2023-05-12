package agents.spencerSchumann;

import java.util.ArrayList;

import engine.core.MarioForwardModel;

/**
 * @author Spencer Schumann
 */
public class Tiles {

    public static final byte EMPTY = 0;
    public static final byte SOLID = 1;
    public static final byte JUMPTHROUGH = 2;
    public static final byte BRICK = 5;
    public static final byte QUESTION = 6;
    public static final byte COIN = 7;
    public static final byte UNKNOWN = -1;

    public static boolean isWall(int tile) {
        switch (tile) {
            case SOLID:
            case BRICK:
            case QUESTION:
                return true;
            default:
                return false;
        }
    }

    private class Column {

        public int startRow = 0;
        int[] tiles = null;

        public void setTile(int y, int tile) {
            if (tiles == null) {
                tiles = new int[1];
                tiles[0] = tile;
                startRow = y;
            } else {
                if (startRow > y) {
                    int expansion = startRow - y;
                    int[] newTiles = new int[tiles.length + expansion];
                    System.arraycopy(tiles, 0, newTiles, expansion, tiles.length);
                    int i;
                    for (i = 0; i < startRow; i++) {
                        newTiles[i] = UNKNOWN;
                    }
                    tiles = newTiles;
                    startRow = y;
                } else if (y >= startRow + tiles.length) {
                    int expansion = y - startRow - tiles.length + 1;
                    int[] newTiles = new int[tiles.length + expansion];
                    System.arraycopy(tiles, 0, newTiles, 0, tiles.length);
                    int i;
                    for (i = tiles.length; i < newTiles.length; i++) {
                        newTiles[i] = UNKNOWN;
                    }
                    tiles = newTiles;
                }
                tiles[y - startRow] = tile;
            }
        }

        public int getTile(int y) {
            if (y < startRow || y >= startRow + tiles.length) {
                return UNKNOWN;
            } else {
                return tiles[y - startRow];
            }
        }
    }

    ArrayList<Column> columns;

    public Tiles() {
        columns = new ArrayList<Column>();
    }

    private void setTile(int x, int y, int tile) {
        if (x < 0) {
            return;
        }
        while (x >= columns.size()) {
            columns.add(null);
        }
        Column c = columns.get(x);
        if (c == null) {
            c = new Column();
            columns.set(x, c);
        }
        c.setTile(y, tile);
    }

    public int getTile(int x, int y) {
        if (x < 0) {
            return EMPTY;
        } else if (x >= columns.size()) {
            return UNKNOWN;
        }
        Column c = columns.get(x);
        if (c == null) {
            return UNKNOWN;
        } else {
            return c.getTile(y);
        }
    }

    public int[][] getScene(int x, int y, int width, int height) {
        int[][] scene = new int[height][width];
        int row, col;
        for (row = 0; row < height; row++) {
            for (col = 0; col < width; col++) {
                scene[row][col] = getTile(col + x, row + y);
            }
        }
        return scene;
    }

    public void addObservation(MarioForwardModel model) {
        int[][] scene = model.getMarioSceneObservation();

        float[] marioPos = model.getMarioFloatPos();
        int offsetX = (int) (marioPos[0] / 16.0f);
        int offsetY = (int) (marioPos[1] / 16.0f);
        offsetX -= model.obsGridWidth / 2;
        offsetY -= model.obsGridHeight / 2;

        int x, y;
        for (x = 0; x < scene.length; x++) {
            for (y = 0; y < scene[x].length; y++) {
                int tile = scene[x][y];
                switch (tile) {
                    case 0: // nothing
                        tile = EMPTY;
                        break;
                    case 17:
                        tile = SOLID;
                        break;
                    case 23: // Brick
                        tile = BRICK;
                        break;
                    case 24: // Question
                        tile = QUESTION;
                        break;
                    case 31: // coin
                        tile = COIN;
                        break;
                    case 59: // Ledge
                        tile = JUMPTHROUGH;
                        break;
                    default:
                        tile = UNKNOWN;
                        break;
                }
                if (tile != UNKNOWN) {
                    setTile(x + offsetX, y + offsetY, tile);
                }
            }
        }
    }
}
