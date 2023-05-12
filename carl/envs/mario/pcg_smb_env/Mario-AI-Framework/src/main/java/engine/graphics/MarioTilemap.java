package engine.graphics;

import java.awt.Graphics;
import java.awt.Image;
import java.util.ArrayList;

import engine.core.MarioGame;
import engine.helper.TileFeature;

public class MarioTilemap extends MarioGraphics {
    public Image[][] sheet;
    public int[][] currentIndeces;
    public int[][] indexShift;
    public float[][] moveShift;
    public int animationIndex;

    public MarioTilemap(Image[][] sheet, int[][] currentIndeces) {
        this.sheet = sheet;
        this.currentIndeces = currentIndeces;
        this.indexShift = new int[currentIndeces.length][currentIndeces[0].length];
        this.moveShift = new float[currentIndeces.length][currentIndeces[0].length];
        this.animationIndex = 0;
    }

    @Override
    public void render(Graphics og, int x, int y) {
        this.animationIndex = (this.animationIndex + 1) % 5;

        int xMin = (x / 16) - 1;
        int yMin = (y / 16) - 1;
        int xMax = (x + MarioGame.width) / 16 + 1;
        int yMax = (y + MarioGame.height) / 16 + 1;

        for (int xTile = xMin; xTile <= xMax; xTile++) {
            for (int yTile = yMin; yTile <= yMax; yTile++) {
                if (xTile < 0 || yTile < 0 || xTile >= currentIndeces.length || yTile >= currentIndeces[0].length) {
                    continue;
                }
                if (this.moveShift[xTile][yTile] > 0) {
                    this.moveShift[xTile][yTile] -= 1;
                    if (this.moveShift[xTile][yTile] < 0) {
                        this.moveShift[xTile][yTile] = 0;
                    }
                }
                ArrayList<TileFeature> features = TileFeature.getTileType(this.currentIndeces[xTile][yTile]);
                if (features.contains(TileFeature.ANIMATED)) {
                    if (this.animationIndex == 0) {
                        this.indexShift[xTile][yTile] = (this.indexShift[xTile][yTile] + 1) % 3;
                    }
                } else {
                    this.indexShift[xTile][yTile] = 0;
                }
                int index = currentIndeces[xTile][yTile] + indexShift[xTile][yTile];
                int move = (int) moveShift[xTile][yTile];
                Image img = sheet[index % 8][index / 8];
                og.drawImage(img, xTile * 16 - x, yTile * 16 - y - move, null);
            }
        }
    }

}
