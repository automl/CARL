package engine.graphics;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsConfiguration;
import java.awt.Image;
import java.awt.Transparency;

import engine.helper.Assets;

public class MarioBackground extends MarioGraphics {
    private Image image;
    private Graphics2D g;
    private int screenWidth;

    public MarioBackground(GraphicsConfiguration graphicsConfiguration, int screenWidth, int[][] indeces) {
        super();
        this.width = indeces[0].length * 16;
        this.height = indeces.length * 16;
        this.screenWidth = screenWidth;

        image = graphicsConfiguration.createCompatibleImage(width, height, Transparency.BITMASK);
        g = (Graphics2D) image.getGraphics();
        g.setComposite(AlphaComposite.Src);

        updateArea(indeces);
    }

    private void updateArea(int[][] indeces) {
        g.setBackground(new Color(0, 0, 0, 0));
        g.clearRect(0, 0, this.width, this.height);
        for (int x = 0; x < indeces[0].length; x++) {
            for (int y = 0; y < indeces.length; y++) {
                int xTile = indeces[y][x] % 8;
                int yTile = indeces[y][x] / 8;
                g.drawImage(Assets.level[xTile][yTile], x * 16, y * 16, 16, 16, null);
            }
        }
    }

    @Override
    public void render(Graphics og, int x, int y) {
        int xOff = x % this.width;
        for (int i = -1; i < this.screenWidth / this.width + 1; i++) {
            og.drawImage(image, -xOff + i * this.width, 0, null);
        }
    }

}
